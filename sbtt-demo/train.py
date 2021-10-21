import pytorch_lightning as pl
from data import LorenzDataModule
from model import SequentialAutoencoder

data_path = '/snel/share/data/lfads_lorenz_20ms/lfads_dataset001.h5'
datamodule = LorenzDataModule(data_path, bandwidth=10)
model = SequentialAutoencoder()
trainer = pl.Trainer(
    callbacks=[
        pl.callbacks.EarlyStopping(monitor="valid_loss", patience=25),
    ],
)
trainer.fit(model, datamodule)
