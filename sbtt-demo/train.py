import torch
import pytorch_lightning as pl
from data import LorenzDataModule
from model import SequentialAutoencoder

data_path = 'lorenz_dataset.h5'

for bandwidth in [None, 25, 20, 15, 10, 5, 3, 2]:
    datamodule = LorenzDataModule(data_path, bandwidth=bandwidth)
    model = SequentialAutoencoder()
    trainer = pl.Trainer(
        callbacks=[
            pl.callbacks.ModelCheckpoint(monitor='valid_loss'),
        ],
        gpus=int(torch.cuda.is_available()),
    )
    trainer.fit(model, datamodule)
