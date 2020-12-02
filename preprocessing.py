#!/usr/bin/env python

import os, sys, math
from collections import Counter
import numpy as np, matplotlib.pyplot as plt
from Bio import SeqIO

# Global constants
alphabet = 'ACDEFGHIKLMNPQRSTVWY'

def find_index(arr, pred):
    """ Find index of the first item that satisfies a predicate """
    for index, elem in enumerate(arr):
        if pred(elem):
            return index
    return -1

def load_uniprot(filepath):
    """ Load all uniprot data from file using Biopython """
    print('Loading uniprot dataset')
    with open(filepath) as handle:
        uniprot = [r for r in SeqIO.parse(handle, 'swiss')]
        repeated_seqs = set(seq for seq, count in Counter(u._seq._data for u in uniprot).items() if count > 1)
    return uniprot, repeated_seqs

def filtered_families(seq_fam, minimum_count = 500, draw_histogram=False):
    """ 
    Filter out sequences belonging to families with less than `minimum_count' examples. 
    Optionally, draw a histogram after filtering to show the distribution of family sizes.
    """
    
    families = Counter(fam for seq, fam in seq_fam.items())
    print('Number of families before filter: {}'.format(len(families)))
    
    filtered_fam = {fam : count for fam, count in families.items() if count >= minimum_count }
    ff_counts = np.array([*filtered_fam.values()])

    if draw_histogram:
        # Draw histogram
        fig, ax = plt.subplots()
        ax.hist(ff_counts)
        ax.set_ylabel('Count')
        ax.set_xlabel('Examples / Family')

    print('Num Examples: {} | Families: {} \n'
          'Mean: {:.2f} | Variance: {:.2f} \n'
          'Min: {} | Max: {}'.format(
            np.sum(ff_counts),
            len(filtered_fam),
            np.mean(ff_counts),
            np.var(ff_counts),
            np.min(ff_counts),
            np.max(ff_counts)))
    
    return filtered_fam

def filter_sequences(parent_directory, filename, label = 'Pfam:'):
    """ Perform all steps of loading, filtering and writing the filtered output to a file """
    uniprot, repeated_seqs = load_uniprot('{}/{}'.format(parent_directory, filename))
    seq_fam = {}
    
    for u in uniprot:
        # Only select those sequences
        #  - With the correct label
        #  - That do not contain illegal characters
        #  - Do not include sequences that are in multiple families
        index = find_index(u.dbxrefs, lambda x: x.startswith(label))
        if index != -1 and len(set(u._seq._data) - set(alphabet)) == 0 and u._seq._data not in repeated_seqs:
            seq_fam[u._seq._data] = u.dbxrefs[index][len(label):]
    
    filtered_fam = filtered_families(seq_fam)
    
    # Write processed output to a separate file
    with open('{}/processed.txt'.format(parent_directory), 'w') as f:
        f.write('\n'.join(' '.join((fam, seq))
                          for seq, fam in seq_fam.items()
                              if fam in filtered_fam))

if __name__ == "__main__":
    dataset_dir = 'Uniprot'
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)
    
    dataset_filename =  'uniprot_sprot.dat'
    file_path = '{}/{}'.format(dataset_dir, dataset_filename)
    if not os.path.exists(file_path):
        tar_file_path = dataset_dir + '/uniprot_sprot-only2019_11.tar.gz'
        if not os.path.exists(tar_file_path):
            from urllib.request import urlretrieve
            # download from:
            link = 'ftp://ftp.uniprot.org/pub/databases/uniprot/previous_releases/release-2019_11/knowledgebase/uniprot_sprot-only2019_11.tar.gz'
            print("Downloading dataset from {}".format(link))
            urlretrieve(link, tar_file_path)

        # unzip .dat file
        import tarfile, gzip, shutil
        
        gz_filename = dataset_filename + '.gz'
        with tarfile.open(tar_file_path, 'r:gz') as f:
            print("Extracting dataset file")
            f.extract(gz_filename, dataset_dir)
            
        gz_file_path = '{}/{}'.format(dataset_dir, gz_filename)
        with gzip.open(gz_file_path) as f_in, open(file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(gz_file_path)
        
    # ----------
    filter_sequences(dataset_dir, dataset_filename)

