#!/usr/bin/env python

from data import load_data, load_data_kfold, alphabet

import sys, os, math, random
from random import randrange, randint
from collections import Counter
import pickle 

import numpy as np

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils import data

# For reproducibility
random.seed(9001)
np.random.seed(9001)
torch.manual_seed(9001)

background_freq = None
def aa_background_freq():
    """ Find background frequencies of amino acids in dataset. """
    global background_freq
    if background_freq is None:
        with torch.no_grad():
            concatenated = torch.cat([torch.sum(one_hot(s, len(alphabet)), 0, keepdim=True) for s in Seq_test])
            total = torch.sum(concatenated, 0)
            print("#AA occurances in training set = {}".format(total))
            background_freq = total / torch.sum(total)
    return background_freq.view(1, len(alphabet), 1)

def one_hot(seq, num_classes, smoothing = None):
    result = torch.zeros((seq.size(0), num_classes), device=seq.device)
    result[torch.arange(seq.size(0)), seq] = 1
    
    if smoothing is not None:
        return (1 - smoothing) * result + smoothing / num_classes
    else:
        return result

def load_to_network_from_file(net, model_file_name):
    net.eval()
    state_dict = torch.load(model_file_name, map_location=torch.device('cpu'))
    net.load_state_dict(state_dict)

def load_new_network_from_file(filename, *args):
    net = ProtClassifier(*args)
    load_to_network_from_file(net, filename)
    return net

def load_motif_detector_weights(net, path):
    """
       This function loads only the convolution weights and biases
       into the network.
       This is necessary because when using torch.save and torch.load
       the model doesn't classify the data correctly. It's as good as
       random. I can only assume that this has to do with the weights
       not being saved or loaded correctly.
    """
    import pickle
    with open(path, 'rb') as f:
        motif_weights_and_biases = pickle.load(f)
    for detector, (weights, biases) in zip(net.motif_detectors, motif_weights_and_biases):
        detector._parameters['weight'].copy_(weights)
        detector._parameters['bias'].copy_(biases)
        
def preprocess_batch(batch):
    x = [one_hot(seq, len(alphabet)).float() for seq in batch]
    x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True).permute(0, 2, 1)
    lengths = torch.tensor([len(s) for s in batch])
    return x, lengths

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, weights = None, smoothing=0.0, dim=-1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.weights   = weights
        self.smoothing = smoothing
        self.dim       = dim

    def forward(self, pred, target, motifs):
        pred = -pred.log_softmax(dim=self.dim)
        
        if self.weights is not None:
            pred = pred * self.weights
        
        with torch.no_grad():
            target = one_hot(target, pred.size(1), smoothing=self.smoothing)
            
        loss = torch.mean(torch.sum(target * pred, dim=self.dim))
        return loss

class ProtClassifier(nn.Module):
    def __init__(self, alphabet_length, num_motifs_of_len, motif_lengths, num_classes, **super_kwargs):
        super().__init__(**super_kwargs)
        
        self.num_classes = num_classes
        self.motif_detectors = nn.ModuleList([
            nn.Conv1d(alphabet_length, num_motifs, motif_length)
            for num_motifs, motif_length in zip(num_motifs_of_len, motif_lengths)
        ])
        
        num_channels = sum(num_motifs_of_len)
        intermediate_layer_size = 150
        
        self.classifier_layers = nn.ModuleList([
            nn.BatchNorm1d(num_channels),
            nn.Linear(num_channels, intermediate_layer_size),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),
            nn.Linear(intermediate_layer_size, num_classes)
        ])
    
    def forward(self, batch, lengths):
        device = self.motif_detectors[0].weight.device
        # This is a length mask for the result of the scan. It's size is the maximum of
        # the output tensors of the motif evaluations. A slice of this is used to mask
        # the output so that the bias doesn't influence the results.
        mask = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(1.0, device=device).repeat(l, max(motif.out_channels for motif in self.motif_detectors))
             for l in lengths]).permute(1, 2, 0)
        motif_evals = []
        
        def one_motif_scan(motif, x):
            x = motif(x)
            x = F.relu(x) * mask[:, :motif.out_channels, motif.kernel_size[0] - 1:]
            motif_presence = (x > 0.0).float()
            denom = torch.sum(motif_presence, 2).float()
            num   = torch.sum(x, 2)
            return num / (denom + 1), motif_presence
        
        motif_evals = [one_motif_scan(motif, batch) for motif in self.motif_detectors]
        presence    = [p for _, p in motif_evals]
        x = [e for e, _ in motif_evals]
        x = torch.cat(x, 1)
        
        for layer in self.classifier_layers:
            x = layer(x)
            
        return x, presence

def fit(net, train_data, test_data,
        num_epochs = 1, batch_size = 64,
        learning_rate = 0.001, learning_rate_decay= 0.50,
        label_smoothing_eps = 0.05,
        save_all_models = False,
        class_weights=None,
        device=torch.device('cpu'),
        parent_directory='Models'):
        
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics       import confusion_matrix, accuracy_score, precision_recall_fscore_support, roc_curve, auc, matthews_corrcoef
    
    # Sequences
    net.to(device)
    
    def collate(data):
        seqs = [seq for seq, fam in data]
        seqs, lens = preprocess_batch(seqs)
        
        fams = torch.stack([fam for seq, fam in data], 0)
        return seqs, lens, fams
    
    extra_loss_args = {} if class_weights is None else {'weights':class_weights.to(device)}
    Seq_train, fam_train = train_data
    train_loss_function  = LabelSmoothingCrossEntropy(smoothing=label_smoothing_eps,
                                                      **extra_loss_args)
    
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, learning_rate_decay)
    
    Seq_test, fam_test = test_data
    test_loss_function = LabelSmoothingCrossEntropy(smoothing = label_smoothing_eps)

    min_loss = float('inf')

    # ========

    min_motif_length = net.motif_detectors[ 0].kernel_size[0]
    max_motif_length = net.motif_detectors[-1].kernel_size[0]

    if not os.path.exists(parent_directory):
        os.mkdir(parent_directory)

    save_folder_name = '{}/{}-{}'.format(parent_directory, min_motif_length, max_motif_length)
    if not os.path.exists(save_folder_name):
        os.mkdir(save_folder_name)

    (loss_history, accuracy_history, precision_history,
     recall_history, f1_score_history, roc_auc_history,
     mcc_history) = [[] for _ in range(7)]

    try:
        for e in range(1, num_epochs + 1):
            # Data Loaders
            
            data_loader_train = data.DataLoader([*zip(Seq_train, fam_train)],
                                                batch_size=batch_size,
                                                num_workers=2,
                                                pin_memory=device.type.startswith('cuda'),
                                                collate_fn=collate)
            data_loader_test  = data.DataLoader([*zip(Seq_test, fam_test)],
                                                batch_size=batch_size,
                                                num_workers=2,
                                                pin_memory=device.type.startswith('cuda'),
                                                collate_fn=collate)
            # -----------------------------
            print("Epoch {} | Motif lengths {} - {}".format(str(e), min_motif_length, max_motif_length))

            # Test first because the model converges very quickly
            save_file_name = '{}/Epoch-{:02d}.pt'.format(save_folder_name, e)
            num_correct = 0
            with torch.no_grad():
                net.eval()
                test_loss = 0
                classifications  = np.array([]).reshape((0, net.num_classes))
                num_test_batches = int(np.ceil(len(Seq_test) / batch_size))

                for batch_index, (batch, lengths, targets) in enumerate(data_loader_test):
                    targets = targets.to(device)
                    preds, presence = net(batch.to(device),lengths)
                    classifications = np.concatenate([classifications, preds.cpu().numpy()])

                    num_correct += torch.sum(targets == torch.argmax(preds, 1)).item()
                    sample_loss  = test_loss_function(preds, targets, net.motif_detectors)
                    test_loss   += sample_loss

                    sys.stdout.write('\rTesting ({}/{}) loss = {:.2e}, approx accuracy = {:.2f}%         '.format(
                        batch_index, num_test_batches, sample_loss,
                        100.0 * num_correct / ((batch_index + 1) * batch_size)))
                    sys.stdout.flush()
                
                y_true   = one_hot(fam_test.cpu(), net.num_classes).numpy()
                y_scores = F.softmax(torch.tensor(classifications), 1).numpy()
                
                y_true_r, y_scores_r = y_true.ravel(), y_scores.ravel()
                fpr, tpr, _ = roc_curve(y_true_r, y_scores_r)
                roc_auc = auc(fpr, tpr)
                
                classifications = np.argmax(classifications, axis=1)
                
                y_true, y_pred = fam_test.cpu().numpy(), classifications
                conf_mat = confusion_matrix(y_true, y_pred, labels=range(net.num_classes))
                np.save('{}/Epoch-{:02d}-ConfusionMatrix'.format(save_folder_name, e), conf_mat)
                
                test_loss /= len(Seq_test)
                accuracy   = accuracy_score(y_true, y_pred)
                precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
                mcc = matthews_corrcoef(y_true, y_pred)

                loss_history.append(test_loss)
                accuracy_history.append(accuracy)
                precision_history.append(precision)
                recall_history.append(recall)
                f1_score_history.append(f1_score)
                roc_auc_history.append(roc_auc)
                mcc_history.append(mcc)
                
                np.save('{}/Epoch-{:02d}-ROC'.format(save_folder_name, e), np.stack((fpr, tpr)))
                
                print(("Epoch {} \"{}\"\n"
                       # "  Loss      = {:.2f}\n"
                       "  Accuracy  = {:.2f}%\n"
                       "  Precision = {:.2f}%\n"
                       "  Recall    = {:.2f}%\n"
                       "  F1-Score  = {:.2f}%\n"
                       "  MCC       = {:.2f}\n"
                       "  ROC AUC   = {:.2f}").format(e, save_file_name,
                                                      # loss,
                                                      100.0 * accuracy,
                                                      100.0 * precision,
                                                      100.0 * recall,
                                                      100.0 * f1_score,
                                                      mcc,
                                                      roc_auc))
            # train
            num_correct = 0
            num_batches = int(np.ceil(len(Seq_train) / batch_size))
            net.train()
            for batch_index, (batch, lengths, targets) in enumerate(data_loader_train):
                targets = targets.to(device)
                optimizer.zero_grad()
                output, presence = net(batch.to(device), lengths)
                loss = train_loss_function(output, targets, net.motif_detectors)
                
                loss.backward()
                optimizer.step()
                
                num_correct += torch.sum(targets == torch.argmax(output, 1)).item()
                sys.stdout.write("\r{}/{} | loss = {:.2e} | approx accuracy = {:.2f}%   ".format(
                    batch_index + 1,
                    num_batches,
                    loss.item(),
                    100.0 * num_correct / ((batch_index + 1) * batch_size) ))
                sys.stdout.flush()
                
            # Note: This measure of accuracy is not correct since the weights get
            # modified after every batch. This is only here to give an idea of
            # how the network is performing on the training set this epoch.
            sys.stdout.write("\rTraining Accuracy = {:.4f}%     \n"
                             .format(100.0 * num_correct / len(Seq_train)))
            scheduler.step()
            
            # Saving weights
            if save_all_models or test_loss < min_loss:
                print("Loss went from {:.2e} to {:.2e} Saving.".format(min_loss, test_loss))
                
                # Network must be in evaluation mode to save batch-norm and dropout layers properly,
                # though the saved file seems to be corrupted anyway
                net.eval()
                torch.save(net.state_dict(), save_file_name)
                min_loss = test_loss
                
        # after test loop
        if not save_all_models:
            torch.save(net.state_dict(), save_file_name)
        
    except KeyboardInterrupt:
        print("\nStopped by User\n")
        
    return np.array([t.cpu().numpy() for t in loss_history]), \
           np.array(accuracy_history), \
           np.array(precision_history), \
           np.array(recall_history), \
           np.array(f1_score_history), \
           np.array(roc_auc_history), \
           np.array(mcc_history)

# Motif Saving
def load_net():
    """
    Load a version of DeePSLiM with default parameters
    """
    net = ProtClassifier(len(alphabet),
                         [5 for _ in range(5, 25)],
                         [*range(5, 25)],
                         len(family_set))
    with torch.no_grad():
        net.eval()
    return net

def extract_pfms(net, seqs:list, batch_size=64, device=torch.device('cpu')):
    """
    Extracts the position frequency matricies for each of the motifs detected in the training data.
    """
    import pickle
    
    # Sequences
    net.to(device)
    
    data_loader = data.DataLoader(seqs,
                                  batch_size=batch_size,
                                  num_workers=2,
                                  pin_memory=device.type.startswith('cuda'),
                                  collate_fn=preprocess_batch)
    
    start_batch_index = 0
    pfm_storage = [torch.zeros((detector.out_channels, detector.in_channels, detector.kernel_size[0]), device=device)
                        for detector in net.motif_detectors]

    try:
        num_correct = 0
        with torch.no_grad():
            net.eval()
            num_test_batches = int(np.ceil(len(seqs) / batch_size))
            
            sys.stdout.write('0/{}'.format(num_test_batches))
            sys.stdout.flush()
            
            for batch_index, (batch, lengths) in enumerate(data_loader):
                if batch_index < start_batch_index:
                    continue
                    
                batch = batch.to(device)
                preds, presence = net(batch, lengths)
                
                min_size = min(detector.kernel_size[0]
                               for detector in net.motif_detectors)
                for motif_index, detection in enumerate(presence):
                    motif_size = motif_index + min_size
                    pfm_motif_size = pfm_storage[motif_index]
                    
                    # ---------------------------------------
                    x = batch.unsqueeze(2)
                    x = F.unfold(x, (1, motif_size))
                    detection = detection.transpose(1, 2)
                    
                    conv = torch.matmul(x, detection)
                    
                    folded = conv.view((conv.size(0), min_size,
                                        len(alphabet), motif_size))
                    folded = torch.sum(folded, 0)
                    # print(pfm_storage, motif_size, folded.size())
                    
                    pfm_storage[motif_index] += folded
                    
                    sys.stdout.write('\r Saving motif of size {:3d}: {:3d}/{:3d}'.format(
                        motif_size, batch_index + 1, num_test_batches))
                    sys.stdout.flush()

                
    except KeyboardInterrupt:
        print("\nStopped by User\n")
    
    return pfm_storage

def save_pfms_meme_format(pfms):
    background_freqs = aa_background_freq().view(-1).numpy()
    ppms = [(p / p.sum(1, keepdim=True)).cpu().detach() for p in pfms]
    
    with open('output_motifs.meme', 'w') as meme:
        meme.write('MEME version 4\n\n')
        meme.write('ALPHABET= {}\n\n'.format("".join(alphabet)))
        meme.write('Background letter frequencies\n')
        for aa, bg_freq in zip(alphabet, background_freqs):
            meme.write("{} {:.6f} ".format(aa, bg_freq))
        meme.write('\n\n')

        motif_lengths = [ppm.size(-1) for ppm in ppms]
        min_size = min(motif_lengths)
        max_size = max(motif_lengths)

        for motif_length, motifs in zip(range(min_size, max_size + 1), ppms):
            motif_prefix = 'letter-probability matrix: alength= {} w= {}\n'.format(len(alphabet), motif_length)
            for index, motif in enumerate(motifs):
                meme.write('MOTIF Length{}-{}\n'.format(motif_length, index))
                meme.write(motif_prefix)
                for row in motif.t():
                    meme.write("".join(["{:.6f} ".format(num) for num in row]) + '\n')
                meme.write('\n')
            

if __name__ == "__main__":
    # x = torch.randint(0, 6, (5,))
    # print(x.dtype)
    # sys.exit(0)

    import argparse
    
    parser = argparse.ArgumentParser()

    # TODO: Make sure that all of these command line options work
    parser.add_argument('--device', type=str, default='cpu',
                        help='The device to use. (ex. cpu, cuda, cuda:0, etc.)')
    parser.add_argument('--num-folds', type=int, default=5,
                        help='The number of folds to use in k-fold cross validation')
    parser.add_argument('--save-location', type=str, default='Base',
                        help='The directory that the neural network will be saved in')
    parser.add_argument('--num-epochs', type=int, default=30,
                        help='The number of epochs to train the network')
    parser.add_argument('--num-motifs-of-length', type=int, default=5,
                        help='The number of motifs of each length')
    parser.add_argument('--min-size', type=int, default=5,
                        help='The size of the smallest convolution filter')
    parser.add_argument('--max-size', type=int, default=24,
                        help='The size of the largest convolution filter')

    args = parser.parse_args()
    
    if args.device.split(':')[0] not in ['cpu', 'cuda']:
        parser.print_help()
        sys.exit(1)
    
    device = torch.device(args.device)
    # preparation for training

    min_motif_length, max_motif_length = args.min_size, args.max_size
    motifs_per_length    = args.num_motifs_of_length
    motif_lengths        = [*range(min_motif_length, max_motif_length + 1)]
    num_motifs_of_length = [motifs_per_length for _ in motif_lengths]
    num_epochs   = args.num_epochs
    parent_dir   = args.save_location

    kfold, sequences, families, \
    class_weights, family_set, \
    family_counts = load_data_kfold(args.num_folds)
    families = torch.tensor(families)

    for kfold_enum, (train_indices, test_indices) in enumerate(kfold):
        save_folder_name = '{}/{}-{}'.format(parent_dir, min_motif_length, max_motif_length) + str(kfold_enum)
        motif_save_location = save_folder_name + '/motif_weights'

        Seq_train = [sequences[i] for i in train_indices]
        Seq_test  = [sequences[i] for i in  test_indices]
        fam_train = families[train_indices]
        fam_test  = families[test_indices]
        _net = ProtClassifier(len(alphabet), num_motifs_of_length,
                              motif_lengths, len(family_set))
        
        if False:
            results = fit(_net, (Seq_train, fam_train), (Seq_test, fam_test),
                          num_epochs=num_epochs,
                          class_weights=class_weights,
                          device=device, parent_directory=parent_dir)
            
            loss_history, accuracy_history, precision_history, recall_history, f1_score_history, roc_auc_history, mcc_history = results 

            save_file_name = save_folder_name + '/Metrics'
            with open(save_file_name + '.txt', 'w') as f:
                f.write("Loss\n")
                f.write(str(loss_history.tolist()))
                f.write("\nAccuracy\n")
                f.write(str(accuracy_history.tolist()))
                f.write("\nPrecision\n")
                f.write(str(precision_history.tolist()))
                f.write("\nRecall\n")
                f.write(str(recall_history.tolist()))
                f.write("\nF1-Score\n")
                f.write(str(f1_score_history.tolist()))
                f.write("\nROC-AUC\n")
                f.write(str(roc_auc_history.tolist()))
                f.write("\nMCC\n")
                f.write(str(mcc_history.tolist()))

            np.save(save_file_name, np.stack((loss_history,
                                              accuracy_history,
                                              precision_history, recall_history,
                                              f1_score_history,  roc_auc_history,
                                              mcc_history)))
        
            # It seems like either torch.save corrupts the data or torch.load doesn't load it properly.
            # Whatever the case, I have to do this to make sure motif detectors are saved...
            with torch.no_grad():
                wbs = [(m._parameters['weight'].clone().cpu(), m._parameters['bias'].clone().cpu())
                       for m in _net.motif_detectors]
                with open(motif_save_location, 'wb') as f:
                    pickle.dump(wbs, f)

        pfms = extract_pfms(_net, Seq_test, device=device)
        save_pfms_meme_format(pfms)

