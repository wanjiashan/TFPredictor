import torch
import torch.utils.data as utils
import numpy as np


def fft_transform(data, n_freq=3):
    # Apply FFT along the temporal dimension and keep top frequencies
    data_fft = torch.fft.rfft(torch.Tensor(data), dim=1)
    frequencies = torch.abs(data_fft)

    # Ensure n_freq does not exceed the frequency dimension size
    n_freq = min(n_freq, frequencies.size(1))

    top_frequencies, _ = torch.topk(frequencies, n_freq, dim=1)
    # Calculate the mean of top frequencies along the temporal dimension
    mean_freq_per_time_segment = top_frequencies.mean(dim=1)
    return mean_freq_per_time_segment.numpy()


def fft_transform_with_timescale(data, timescale, n_freq=3):
    # Apply FFT along the temporal dimension and keep top frequencies
    data_fft = torch.fft.rfft(torch.Tensor(data), dim=1)
    frequencies = torch.abs(data_fft)

    # Ensure n_freq does not exceed the frequency dimension size
    n_freq = min(n_freq, frequencies.size(1))

    top_frequencies, _ = torch.topk(frequencies, n_freq, dim=1)
    # Calculate the mean of top frequencies along the temporal dimension
    top_frequencies_mean = top_frequencies.mean(dim=1)

    # Segment the mean frequencies by the given timescale
    segmented_means = []
    num_segments = data.shape[1] // timescale
    for i in range(num_segments):
        segment_mean = top_frequencies_mean[:, i * timescale:(i + 1) * timescale].mean(dim=1)
        segmented_means.append(segment_mean)

    # If there's remaining data that doesn't fit into the timescale segments
    if data.shape[1] % timescale != 0:
        remaining_mean = top_frequencies_mean[:, num_segments * timescale:].mean(dim=1)
        segmented_means.append(remaining_mean)

    return torch.stack(segmented_means, dim=1).numpy()


def PrepareDataset(speed_matrix, BATCH_SIZE=12, seq_len=5, pred_len=5, train_propotion=0.8, valid_propotion=0.1, timescale=48):
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

    # Convert to numpy arrays
    speed_sequences = np.array(speed_sequences)
    speed_labels = np.array(speed_labels)

    # Apply FFT and transform data
    fft_features = fft_transform_with_timescale(speed_sequences, timescale)
    # Expand dimensions to concatenate with the original sequences
    fft_features_expanded = np.expand_dims(fft_features, axis=1).repeat(seq_len, axis=1)
    # Concatenate along the feature axis, which is the last axis (axis=2)
    speed_sequences = np.concatenate((speed_sequences, fft_features_expanded), axis=2)

    # Reduce the feature dimension to 307 if necessary
    if speed_sequences.shape[2] > 883:
        speed_sequences = speed_sequences[:, :, :883]

    # Reshape labels to match sequences
    speed_labels = speed_labels.reshape(speed_labels.shape[0], seq_len, -1)

    # Shuffle & split the dataset
    sample_size = speed_sequences.shape[0]
    index = np.arange(sample_size)
    np.random.shuffle(index)
    train_index = int(sample_size * train_propotion)
    valid_index = int(sample_size * (train_propotion + valid_propotion))

    # Splitting data
    train_data, train_label = speed_sequences[:train_index], speed_labels[:train_index]
    valid_data, valid_label = speed_sequences[train_index:valid_index], speed_labels[train_index:valid_index]
    test_data, test_label = speed_sequences[valid_index:], speed_labels[valid_index:]

    # Convert to tensors
    train_data, train_label = torch.Tensor(train_data), torch.Tensor(train_label)
    valid_data, valid_label = torch.Tensor(valid_data), torch.Tensor(valid_label)
    test_data, test_label = torch.Tensor(test_data), torch.Tensor(test_label)

    # Create datasets and dataloaders
    train_dataset = utils.TensorDataset(train_data, train_label)
    valid_dataset = utils.TensorDataset(valid_data, valid_label)
    test_dataset = utils.TensorDataset(test_data, test_label)
    train_dataloader = utils.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    valid_dataloader = utils.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_dataloader = utils.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    return train_dataloader, valid_dataloader, test_dataloader, max_speed