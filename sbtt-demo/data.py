import h5py
import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader


def mask_data(data, bandwidth, rng):
    nan_mask = np.full(data.shape, np.nan)
    for i, sample in enumerate(data):
        for j, timestep in enumerate(sample):
            neuron_ixs = np.arange(len(timestep))
            sampled_ixs = rng.choice(neuron_ixs, size=bandwidth, replace=False)
            nan_mask[i, j, sampled_ixs] = 1.0
    return data * nan_mask


class LorenzDataModule(pl.LightningDataModule):
    def __init__(self, data_path, bandwidth=None, batch_size=64, num_workers=4, seed=0):
        super().__init__()
        self.save_hyperparameters()
        self.rng = np.random.RandomState(seed=seed)
    
    def setup(self, stage=None):
        hps = self.hparams
        # Load data arrays from file
        with h5py.File(hps.data_path, 'r') as h5file:
            data_dict = {k: v[()] for k, v in h5file.items()}
        train_spikes = data_dict['train_data']
        valid_spikes = data_dict['valid_data']
        train_rates = data_dict['train_truth']
        valid_rates = data_dict['valid_truth']
        # Simulate bandwidth-limited sampling
        if hps.bandwidth is not None:
            train_spikes = mask_data(train_spikes, hps.bandwidth, self.rng)
            valid_spikes = mask_data(valid_spikes, hps.bandwidth, self.rng)
        # Convert data to Tensors
        train_spikes = torch.tensor(train_spikes, dtype=torch.float)
        valid_spikes = torch.tensor(valid_spikes, dtype=torch.float)
        train_rates = torch.tensor(train_rates, dtype=torch.float)
        valid_rates = torch.tensor(valid_rates, dtype=torch.float)
        # Store datasets
        self.train_ds = TensorDataset(train_spikes, train_rates)
        self.valid_ds = TensorDataset(valid_spikes, valid_rates)

    def train_dataloader(self):
        train_dl = DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
        )
        return train_dl
    
    def val_dataloader(self):
        valid_dl = DataLoader(
            self.valid_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )
        return valid_dl
