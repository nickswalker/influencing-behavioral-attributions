import logging

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import TensorDataset, DataLoader

from models import mdn
from models.mdn import sample_mog_n
from models.plotting import make_mog, make_density
from models.util import LitProgressBar
import numpy as np


class LitMDN(pl.LightningModule):

    def __init__(self, in_features, out_features, hidden_size, gaussians):
        super().__init__()
        self.save_hyperparameters()
        self.features = nn.Sequential(nn.Linear(in_features, hidden_size), nn.Tanh())
        self.module_list = nn.ModuleList([mdn.MDN(hidden_size, 1, gaussians) for _ in range(out_features)])

    def forward(self, x):
        feats = self.features(x)
        out = [m(feats) for m in self.module_list]
        pi = torch.stack([p[0] for p in out], dim=1)
        sigma = torch.stack([p[1] for p in out], dim=1)
        mu = torch.stack([p[2] for p in out], dim=1)
        return pi, sigma, mu

    def training_step(self, batch, batch_idx):
        x, y = batch
        feats = self.features(x)
        out = [m(feats) for m in self.module_list]
        loss = [mdn.mdn_loss(*params, y[:, i].unsqueeze(1)) for i, params in enumerate(out)]
        loss = torch.stack(loss).sum()
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        feats = self.features(x)
        out = [m(feats) for m in self.module_list]
        loss = [mdn.mdn_loss(*params, y[:, i]) for i, params in enumerate(out)]
        loss = torch.stack(loss).sum()
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        feats = self.features(x)
        out = [m(feats) for m in self.module_list]
        loss = [mdn.mdn_loss(*params, y[:, i]) for i, params in enumerate(out)]
        sum_loss = torch.stack(loss).sum()
        self.log('test_loss', sum_loss)
        for i in range(len(loss)):
            self.log(f'test_loss_{i}', loss[i])
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items["val_loss"] = f"{self.trainer.logged_metrics['val_loss']:.3g}"
        return items


def train_mdn(x, y, x_val, y_val, x_test, y_test, name=None):
    if name is None:
        name = "mdn"
    # initialize the model
    module = LitMDN(in_features=11, out_features=3, hidden_size=11, gaussians=3)
    y = y[["factor0", "factor1", "factor2"]].to_numpy()
    y_val = y_val[["factor0", "factor1", "factor2"]].to_numpy()
    y_test = y_test[["factor0", "factor1", "factor2"]].to_numpy()

    # configure logging at the root level of lightning
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    import warnings
    warnings.filterwarnings("ignore")
    train_data = TensorDataset(torch.from_numpy(np.vstack(x["features"])).float(), torch.from_numpy(y).float())
    val_data = TensorDataset(torch.from_numpy(np.vstack(x_val["features"])).float(), torch.from_numpy(y_val).float())
    test_data = TensorDataset(torch.from_numpy(np.vstack(x_test["features"])).float(), torch.from_numpy(y_test).float())
    checkpoint_callback = ModelCheckpoint(monitor='val_loss')
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=5000,
        verbose=False,
        mode='min'
    )
    progress = LitProgressBar()
    # progress_bar_refresh_rate=0
    logger = TensorBoardLogger('train_logs', name=name)
    trainer = pl.Trainer(deterministic=True, max_epochs=2000,
                         callbacks=[checkpoint_callback, early_stop_callback, progress], val_check_interval=1.0, logger=logger)
    trainer.fit(module, DataLoader(train_data, batch_size=32, shuffle=True), DataLoader(val_data))
    best_validation = LitMDN.load_from_checkpoint(checkpoint_path=checkpoint_callback.best_model_path)
    # trainer.test(module, DataLoader(test_data), verbose=False)[0]
    results = trainer.test(best_validation, DataLoader(test_data), verbose=False)[0]

    # sample new points from the trained model
    pi, sigma, mu = best_validation.forward(test_data.tensors[0])
    samples = torch.zeros([len(y_test), 1000, 3])
    samples[:, :, 0] = sample_mog_n(1000, pi[:, 0, :], sigma[:, 0, :], mu[:, 0, :]).squeeze()
    samples[:, :, 1] = sample_mog_n(1000, pi[:, 1, :], sigma[:, 1, :], mu[:, 1, :]).squeeze()
    samples[:, :, 2] = sample_mog_n(1000, pi[:, 2, :], sigma[:, 2, :], mu[:, 2, :]).squeeze()

    import pandas as pd
    for i in range(len(x_test)):
        truth_i = x_test["uuid"] == x_test["uuid"].iloc[i]
        truth_y = y_test[truth_i]
        # fig = make_density("test", pd.DataFrame(pd.DataFrame(samples[0].detach().numpy())), true_points=truth_y)
        # fig.show()
        fig = make_mog(f"{name} traj={x_test['uuid'].iloc[i]}", pi[i], sigma[i], mu[i], true_points=truth_y)
        fig.show()
    return results
