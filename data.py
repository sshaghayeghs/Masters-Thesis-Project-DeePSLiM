from collections import Counter
import numpy as np, torch

# This file defines the alphabet used in training and 

alphabet = 'ACDEFGHIKLMNPQRSTVWY'

# Load data
def load_data():
    print("Loading Data")
    
    all_data = np.array([x.split() for x in open('Uniprot/processed.txt')])
    families, sequences = all_data[:, 0], all_data[:, 1]
    print("Loaded")
    
    # Convert data and measure size of families
    alphabet_indices = { letter : index for index, letter in enumerate(alphabet) }
    sequences  = [torch.tensor([alphabet_indices[aa] for aa in seq]) for seq in sequences]
    family_set = set(families)
    print("    {} sequences\n    {} families".format(len(sequences), len(family_set)))
    fam2num    = {f:i for i, f in enumerate(family_set)}
    num2fam    = {i:f for i, f in fam2num.items()}
    families   = [fam2num[f] for f in families]
    print("Organized")
    
    from sklearn.model_selection import train_test_split
    Seq_train, Seq_test, fam_train, fam_test = train_test_split(sequences, families, test_size = 1.0/5.0)
    fam_train, fam_test = torch.tensor(fam_train), torch.tensor(fam_test)
    
    print("Train size = {} Test size = {}".format(len(Seq_train), len(Seq_test)))
    print("Split")

    family_counts = Counter(families)
    class_weights = torch.tensor([1.0/family_counts[i] for i in range(len(family_counts))])
    
    return Seq_train, Seq_test, fam_train, fam_test, class_weights, family_set, family_counts

if __name__ == "__main__":
    Seq_train, Seq_test, fam_train, fam_test, class_weights, family_set, family_counts = load_data()
    # TODO print some info about the data loaded
