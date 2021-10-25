import os
import torch
import pandas as pd
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from glob import glob

from data import LorenzDataModule
from model import SequentialAutoencoder

TOTAL_OBS = 29
model_data = {
    29: 'lightning_logs/version_0',
    25: 'lightning_logs/version_1',
    20: 'lightning_logs/version_2',
    15: 'lightning_logs/version_3',
    10: 'lightning_logs/version_4',
    5: 'lightning_logs/version_5',
    3: 'lightning_logs/version_6',
    2: 'lightning_logs/version_7',
}
results = []
for bandwidth, model_dir in model_data.items():
    # Load the model
    ckpt_pattern = os.path.join(model_dir, 'checkpoints/*.ckpt')
    ckpt_path = sorted(glob(ckpt_pattern))[0]
    model = SequentialAutoencoder.load_from_checkpoint(ckpt_path)
    # Load the data
    data_path = 'lorenz_dataset.h5'
    datamodule = LorenzDataModule(data_path, bandwidth=bandwidth)
    # Create a trainer
    trainer = pl.Trainer(logger=False, gpus=int(torch.cuda.is_available()))
    result = trainer.validate(model, datamodule)[0]
    result['drop_ratio'] = 1 - bandwidth / TOTAL_OBS
    results.append(result)
    # Plot examples
    fig, axes = plt.subplots(nrows=5, ncols=3, sharex=True, sharey=True)
    dataloader = datamodule.val_dataloader()
    for i, (ax_row, batch) in enumerate(zip(axes, dataloader)):
        valid_spikes, valid_truth = batch
        valid_logrates = model(valid_spikes)
        valid_rates = torch.exp(valid_logrates).detach().numpy() * 0.05
        # Plot just the first sample from each batch
        ax_row[0].imshow(valid_spikes[0].T)
        ax_row[1].imshow(valid_truth[0].T)
        ax_row[2].imshow(valid_rates[0].T)
    fig.suptitle(f'rate recovery, bandwidth: {bandwidth}')
    plt.tight_layout()
results = pd.DataFrame(results)
plt.figure(figsize=(3, 2.5))
plt.plot(results.drop_ratio, results.valid_r2, marker='o', color='slateblue')
plt.xlim(-0.1, 1.1)
plt.ylim(0, 1)
plt.xlabel('Fraction dropped samples')
plt.ylabel('Match to true rates ($R^2$)')
plt.grid()
plt.tight_layout()
plt.savefig('result.png')
plt.show()
