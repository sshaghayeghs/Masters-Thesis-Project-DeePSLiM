import os, sys, math
import numpy as np
import shutil
from data import alphabet, load_data
from train import fit, ProtClassifier
import torch

def grid_search_motifs(min_motifs, max_motifs, device=torch.device('cpu'), num_epochs=30):
    total_motifs = (max_motifs - min_motifs + 1) * 5
    
    Seq_train, Seq_test, fam_train, fam_test, class_weights, family_set, family_counts = load_data()
    
    parent_directory = 'Grid-Search'
    if not os.path.exists(parent_directory):
        os.mkdir(parent_directory)
    
    for min_len in range(min_motifs, max_motifs+2):
        for max_len in range(min_len+1, max_motifs+2):
            motif_lengths = [*range(min_len, max_len)]
            motifs_per_length, remainder = divmod(total_motifs, len(motif_lengths))
            num_motifs_of_length = [motifs_per_length + (0 if i < (len(motif_lengths) - remainder) else 1)
                                        for i, _ in enumerate(motif_lengths)]
            
            print("Training for motif lengths: {} - {}".format(min_len, max_len-1))
            net = ProtClassifier(len(alphabet), num_motifs_of_length, motif_lengths, len(family_set))
            
            results = fit(net,
                          (Seq_train, fam_train),
                          (Seq_test,  fam_test),
                          num_epochs       = num_epochs,
                          class_weights    = class_weights,
                          device           = device,
                          parent_directory = parent_directory)
            loss_history, accuracy_history, precision_history, \
            recall_history, f1_score_history, roc_auc_history, \
                mcc_history = results 

            save_file_name = '{}/{}-{}'.format(parent_directory, min_len, max_len-1)
            if not os.path.exists(save_file_name):
                os.mkdir(save_file_name)
                
            save_file_name = save_file_name + '/Metrics'
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

            np.save(save_file_name, np.stack((loss_history,      accuracy_history,
                                              precision_history, recall_history,
                                              f1_score_history,  roc_auc_history,
                                              mcc_history)))
            
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu',
                        help='The device to use. (ex. cpu, cuda, cuda:0, etc.)')
    args = parser.parse_args()
    
    if args.device.split(':')[0] not in ['cpu', 'cuda']:
        parser.print_help()
        sys.exit(1)
    
    device = torch.device(args.device)
    
    grid_search_motifs(5, 24, device=device)



