import torch
import numpy as np
import pytorch_lightning as pl
from torch import nn
from sklearn.metrics import r2_score

class SequentialAutoencoder(pl.LightningModule):
    def __init__(self, 
                 input_size=29, 
                 hidden_size=50,
                 learning_rate=1e-3,
                 weight_decay=1e-4,
                 rate_conversion_factor=0.05):
        super().__init__()
        self.save_hyperparameters()
        # Instantiate bidirectional GRU encoder
        self.encoder = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        # Instantiate linear mapping to initial conditions
        self.ic_linear = nn.Linear(2*hidden_size, hidden_size)
        # Instantiate autonomous GRU decoder
        self.decoder = nn.GRU(
            input_size=1, # Not used
            hidden_size=hidden_size,
            batch_first=True,
        )
        # Instantiate linear readout
        self.readout = nn.Linear(
            in_features=hidden_size,
            out_features=input_size,
        )

    def forward(self, x):
        # Interpolate NaNs with zeros
        x = torch.nan_to_num(x, nan=0.0)
        # Pass data through the model
        _, h_n = self.encoder(x)
        # Combine output from fwd and bwd encoders
        h_n = torch.cat([*h_n], -1)
        # Compute initial condition
        ic = self.ic_linear(h_n)
        # Create an empty input tensor
        input_placeholder = torch.zeros_like(x)[:, :, :1]
        # Unroll the decoder
        latents, _ = self.decoder(input_placeholder, torch.unsqueeze(ic, 0))
        # Map decoder state to logrates
        logrates = self.readout(latents)
        return logrates

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.hparams.learning_rate, 
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer
    
    def training_step(self, batch, batch_ix):
        x, truth = batch
        # Keep track of location of observed data
        mask = ~torch.isnan(x)
        # Pass data through the model
        logrates = self.forward(x)
        # Mask unobserved steps
        x_obs = torch.masked_select(x, mask)
        logrates_obs = torch.masked_select(logrates, mask)
        # Compute Poisson log-likelihood
        loss = nn.functional.poisson_nll_loss(logrates_obs, x_obs)
        self.log('train_loss', loss, on_epoch=True)
        self.log('train_nll', loss, on_epoch=True)
        # Compute match to true rates
        truth = truth.detach().numpy()
        rates = torch.exp(logrates).detach().numpy() 
        rates *= self.hparams.rate_conversion_factor
        truth = np.concatenate([*truth])
        rates = np.concatenate([*rates])
        r2 = r2_score(truth, rates)
        self.log('train_r2', r2, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_ix):
        x, truth = batch
        # Keep track of location of observed data
        mask = ~torch.isnan(x)
        # Pass data through the model
        logrates = self.forward(x)
        # Mask unobserved steps
        x_obs = torch.masked_select(x, mask)
        logrates_obs = torch.masked_select(logrates, mask)
        # Compute Poisson log-likelihood
        loss = nn.functional.poisson_nll_loss(logrates_obs, x_obs)
        self.log('valid_loss', loss, on_epoch=True)
        self.log('valid_nll', loss, on_epoch=True)
        truth = truth.detach().numpy()
        rates = torch.exp(logrates).detach().numpy() 
        rates *= self.hparams.rate_conversion_factor
        truth = np.concatenate([*truth])
        rates = np.concatenate([*rates])
        r2 = r2_score(truth, rates)
        self.log('valid_r2', r2, on_epoch=True)
        return loss
