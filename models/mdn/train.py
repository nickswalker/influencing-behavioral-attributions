import logging

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import TensorDataset, DataLoader

from models.mdn.lit_mdn import LitMDN
from models.util import LitProgressBar
import numpy as np


def collate_fn_padd(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    lengths = torch.tensor([ x.shape[0] for x, _ in batch])

    x = [ torch.Tensor(x) for x, y in batch]
    y = torch.stack([y for _, y in batch])
    x = torch.nn.utils.rnn.pad_sequence(x, padding_value=-3)

    mask = (x != -3)
    return x, lengths, mask, y

default_hparam = {"hidden_size": 5, "gaussians": 4, "noise_regularization": 0.15, "early_stopping":200, "epochs":10000}
def train_mdn(x, y, x_val, y_val, x_test, y_test, hparams={}, name=None):
    hparams = { **default_hparam, **hparams}
    if name is None:
        name = "mdn"
    # initialize the model
    module = LitMDN(in_features=11, out_features=3, hidden_size=hparams["hidden_size"], gaussians=hparams["gaussians"], noise_regularization=hparams["noise_regularization"])
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
        patience=hparams["early_stopping"],
        verbose=False,
        mode='min'
    )
    progress = LitProgressBar()
    # progress_bar_refresh_rate=0
    logger = TensorBoardLogger('train_logs', name=name)
    trainer = pl.Trainer(terminate_on_nan=True, deterministic=True, max_epochs=hparams["epochs"],
                         callbacks=[checkpoint_callback, early_stop_callback, progress], val_check_interval=1.0, logger=logger)
    i = 0
    while True:
        try:
            trainer.fit(module, DataLoader(train_data, batch_size=32, shuffle=True), DataLoader(val_data))
            break
        except ValueError as e:
            i += 1
            print(e)
            print(f"Retrying {i}...")

    # trainer.test(module, DataLoader(test_data), verbose=False)[0]
    results = trainer.test(None, DataLoader(test_data, batch_size=len(test_data)), ckpt_path="best", verbose=False)[0]
    results = {**trainer.logged_metrics, **results}
    best_validation = LitMDN.load_from_checkpoint(checkpoint_path=checkpoint_callback.best_model_path)
    return results, best_validation, checkpoint_callback.best_model_path
