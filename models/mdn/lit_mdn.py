import pytorch_lightning as pl
import torch
from torch import nn as nn

from models.mdn import mdn


class LitMDN(pl.LightningModule):

    def __init__(self, in_features, out_features, hidden_size, gaussians, noise_regularization=0.0):
        super().__init__()
        self.save_hyperparameters()
        self.noise_regularization = noise_regularization
        self.features = nn.Sequential(nn.Linear(in_features, hidden_size), nn.Tanh())
        self.mdn = mdn.MDN(hidden_size, out_features, gaussians)

    def log_prob(self, x, y):
        feats = self.features(x)
        return self.mdn.log_prob(feats, y)

    def forward(self, x):
        feats = self.features(x)
        return self.mdn(feats)

    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.noise_regularization != 0.0:
            x += torch.normal(0, self.noise_regularization, x.shape, device=self.device)
            y += torch.normal(0, self.noise_regularization, y.shape, device=self.device)
        feats = self.features(x)
        loss = torch.mean(self.mdn.nll(feats, y))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        feats = self.features(x)
        loss = torch.mean(self.mdn.nll(feats, y))
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        feats = self.features(x)
        nll = self.mdn.nll(feats, y)
        loss = torch.mean(nll)
        self.log('test_loss', loss)
        self.log('test_loss_std', torch.std(nll))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        # Our model is really small, but LBFGS is unstable for its loss landscape
        #optimizer = torch.optim.LBFGS(self.parameters())
        return optimizer

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items["val_loss"] = f"{self.trainer.logged_metrics['val_loss']:.3g}"
        return items