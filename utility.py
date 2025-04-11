import os
import shutil
import random
import pandas as pd
from variables import data_dir, train_dir

def create_subset(input_dir, output_dir, subset_size=1000):
    """
    input_dir: directory containing the original dataset
    output_dir: directory to save the subset
    subset_size: number of samples to include in the subset
    """
    if os.path.exists(output_dir):
        print(f"Warning: {output_dir} already exists. Deleting it.")
        shutil.rmtree(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get all sample directories
    sample_dirs = [os.path.join(input_dir, d) for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    # Randomly select a subset of sample directories
    selected_samples = random.sample(sample_dirs, min(subset_size, len(sample_dirs)))

    for sample in selected_samples:
        dest_sample_dir = os.path.join(output_dir, os.path.basename(sample))
        shutil.copytree(sample, dest_sample_dir)
        print(f"Copied {sample} to {dest_sample_dir}")

def check_size():
    df = pd.read_csv('../../Datasets/20BN_Jester_Dataset/Train.csv')
    total = len(df)* 37
    print('train:', len(df)* 37)
    print('train:', len(df))

    df = pd.read_csv('../../Datasets/20BN_Jester_Dataset/Test.csv')
    total += len(df)* 37
    print('test:', len(df)*37)
    print('test:', len(df))

    df = pd.read_csv('../../Datasets/20BN_Jester_Dataset/Validation.csv')
    total += len(df)* 37
    print('val:', len(df)*37)
    print('val:', len(df))

    print('total:', total)


