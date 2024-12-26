import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse
import torch.utils.data as utils
def fft_transform(data, n_freq=3):
    data_fft = torch.fft.rfft(torch.Tensor(data), dim=1)
    frequencies = torch.abs(data_fft)
    n_freq = min(n_freq, frequencies.size(1))
    top_frequencies, _ = torch.topk(frequencies, n_freq, dim=1)
    mean_freq_per_time_segment = top_frequencies.mean(dim=1)
    return mean_freq_per_time_segment.numpy()

def fft_transform_with_timescale(data, timescale, n_freq=3):
    data_fft = torch.fft.rfft(torch.Tensor(data), dim=1)
    frequencies = torch.abs(data_fft)
    n_freq = min(n_freq, frequencies.size(1))
    top_frequencies, _ = torch.topk(frequencies, n_freq, dim=1)
    top_frequencies_mean = top_frequencies.mean(dim=1)
    segmented_means = []
    num_segments = data.shape[1] // timescale
    for i in range(num_segments):
        segment_mean = top_frequencies_mean[:, i * timescale:(i + 1) * timescale].mean(dim=1)
        segmented_means.append(segment_mean)
    if data.shape[1] % timescale != 0:
        remaining_mean = top_frequencies_mean[:, num_segments * timescale:].mean(dim=1)
        segmented_means.append(remaining_mean)
    return torch.stack(segmented_means, dim=1).numpy()

def PrepareDataset(speed_matrix, BATCH_SIZE=-1, seq_len=5, pred_len=5, train_propotion=0.8, valid_propotion=0.1, timescale=48):
    time_len = speed_matrix.shape[0]
    max_speed = speed_matrix.max().max()
    min_speed = speed_matrix.min().min()
    speed_matrix = (speed_matrix - min_speed) / (max_speed - min_speed)

    speed_sequences, speed_labels = [], []
    for i in range(time_len - seq_len - pred_len):
        seq = speed_matrix.iloc[i:i + seq_len].values
        label = speed_matrix.iloc[i + seq_len:i + seq_len + pred_len].values
        speed_sequences.append(seq)
        speed_labels.append(label)

    speed_sequences = np.array(speed_sequences)
    print(f"speed_sequences shape before model input: {speed_sequences.shape}")

    speed_labels = np.array(speed_labels)

    fft_features = fft_transform_with_timescale(speed_sequences, timescale)
    fft_features_expanded = np.expand_dims(fft_features, axis=1).repeat(seq_len, axis=1)
    speed_sequences = np.concatenate((speed_sequences, fft_features_expanded), axis=2)

    if speed_sequences.shape[2] > 307:
        speed_sequences = speed_sequences[:, :, :307]

    speed_labels = speed_labels.reshape(speed_labels.shape[0], pred_len, -1)

    sample_size = speed_sequences.shape[0]
    index = np.arange(sample_size)
    np.random.shuffle(index)
    train_index = int(sample_size * train_propotion)
    valid_index = int(sample_size * (train_propotion + valid_propotion))

    train_data, train_label = speed_sequences[:train_index], speed_labels[:train_index]
    valid_data, valid_label = speed_sequences[train_index:valid_index], speed_labels[train_index:valid_index]
    test_data, test_label = speed_sequences[valid_index:], speed_labels[valid_index:]

    train_data, train_label = torch.Tensor(train_data), torch.Tensor(train_label)
    valid_data, valid_label = torch.Tensor(valid_data), torch.Tensor(valid_label)
    test_data, test_label = torch.Tensor(test_data), torch.Tensor(test_label)

    train_dataset = utils.TensorDataset(train_data, train_label)
    valid_dataset = utils.TensorDataset(valid_data, valid_label)
    test_dataset = utils.TensorDataset(test_data, test_label)
    train_dataloader = utils.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    valid_dataloader = utils.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_dataloader = utils.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    return train_dataloader, valid_dataloader, test_dataloader, max_speed
