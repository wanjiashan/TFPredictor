import argparse
import numpy as np
import pandas as pd
from prepare import PrepareDataset
from train_STGmamba import TrainSTG_Mamba, TestSTG_Mamba
from train_rnn import TrainLSTM, TestLSTM


parser = argparse.ArgumentParser(description='Train & Test STG_Mamba for traffic/weather/flow forecasting')
# choose dataset
parser.add_argument('-dataset', type=str, default='pems04', help='which dataset to run [options: know_air, pems04, hz_metro]')
# choose model
parser.add_argument('-model', type=str, default='STGmamba', help='which model to train & test [options: STGmamba, lstm]')
# choose number of node features
parser.add_argument('-mamba_features', type=int, default=307, help='number of features for the STGmamba model [options: 307,184,80]')

args = parser.parse_args()

###### loading data #######

if args.dataset == 'know_air':
    print("\nLoading KnowAir Dataset...")
    speed_matrix = pd.read_csv('Know_Air/knowair_temperature.csv', sep=',')
    A = np.load('Know_Air/knowair_adj_mat.npy')

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
    speed_matrix = pd.read_csv('PEMS07/pems07_flow (1).csv', sep=',')
    A = np.load('PEMS07/PEMS07-dtw-288-1- (2).npy')
print("\nPreparing train/test data...")
train_dataloader, valid_dataloader, test_dataloader, max_value = PrepareDataset(speed_matrix, BATCH_SIZE=48)
print(f"Train dataloader length: {len(train_dataloader)}")
print(f"Validation dataloader length: {len(valid_dataloader)}")
print(f"Test dataloader length: {len(test_dataloader)}")

# models you want to use
# models you want to use
if args.model == 'STGmamba':
    print("\nTraining STGmamba model...")
    STGmamba, STGmamba_loss = TrainSTG_Mamba(train_dataloader, valid_dataloader, A, K=3, num_epochs=200, mamba_features=args.mamba_features)
    print("\nTesting STGmamba model...")
    results = TestSTG_Mamba(STGmamba, test_dataloader, max_value)

elif args.model == 'lstm':
    print("\nTraining lstm model...")
    lstm, lstm_loss = TrainLSTM(train_dataloader, valid_dataloader, num_epochs=200)
    print("\nTesting lstm model...")
    results = TestLSTM(lstm, test_dataloader, max_value)
