import argparse
import numpy as np
import pandas as pd
from prepare import PrepareDataset
from train_TFPredictor import TrainTFPredictor, TestTFPredictor
from train_rnn import TrainLSTM, TestLSTM
import matplotlib.pyplot as plt



parser = argparse.ArgumentParser(description='Train & Test TFPredictor for traffic/weather/flow forecasting')
# choose dataset
parser.add_argument('-dataset', type=str, default='pems04', help='which dataset to run [options: pems04,PEMSBY,metr-la]')
# choose model
parser.add_argument('-model', type=str, default='TFPredictor', help='which model to train & test [options: TFPredictor.py]')
# choose number of node features
parser.add_argument('-mamba_features', type=int, default=307, help='number of features for the STGmamba model [options: 307,325,207]')

args = parser.parse_args()

###### loading data #######

if args.dataset == 'PEMSBY':
    print("\nLoading pems08 data...")
    speed_matrix = pd.read_csv('PEMSBY/pemsby_flow.csv', sep=',')
    A = np.load('PEMSBY/pemsby-dtw-288-1-.npy')

elif args.dataset == 'pems04':
    print("\nLoading PEMS04 data...")
    speed_matrix = pd.read_csv('PEMS04/pems04_flow.csv', sep=',')
    A = np.load('PEMS04/pems04_adj.npy')

elif args.dataset == 'PEMS08':
    print("\nLoading pems08 data...")
    speed_matrix = pd.read_csv('PEMS08/pems08_flow.csv', sep=',')
    A = np.load('PEMS08/PEMS08-dtw-288-1-.npy')

elif args.dataset == 'PEMS03':
    print("\nLoading HZ-Metro data...")
    speed_matrix = pd.read_csv('PEMS03/pems03_flow.csv', sep=',')
    A = np.load('PEMS03/PEMS03-dtw-288-1-.npy')
elif args.dataset == 'PEMS07':
    print("\nLoading HZ-Metro data...")
    speed_matrix = pd.read_csv('PEMS07/pems07_flow.csv', sep=',')
    A = np.load('PEMS07/PEMS07-dtw-288-1-.npy')

elif args.dataset == 'metr-la':
    print("\nLoading metr-la data...")
    speed_matrix = pd.read_csv('metr-la/metr-la_flow.csv', sep=',')
    A = np.load('metr-la/metr-la-dtw-288-1-.npy')
print("\nPreparing train/test data...")
train_dataloader, valid_dataloader, test_dataloader, max_value = PrepareDataset(speed_matrix, BATCH_SIZE=48)
print(f"Train dataloader length: {len(train_dataloader)}")
print(f"Validation dataloader length: {len(valid_dataloader)}")
print(f"Test dataloader length: {len(test_dataloader)}")

if args.model == 'TFPredictor':
    print("\nTraining TFPredictor model...")
    TFPredictor, TFPredictor_loss = TrainTFPredictor(train_dataloader, valid_dataloader, A, K=3, num_epochs=200, mamba_features=args.mamba_features)
    print("\nTesting TFPredictor model...")
    results = TestTFPredictor(TFPredictor, test_dataloader, max_value)
