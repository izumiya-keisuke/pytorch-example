"""
Copyright 2022 Keisuke Izumiya

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import Optional

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from pytorch_lightning import LightningModule, Trainer
from torch import Tensor
from torchvision.datasets import MNIST

from ..data import make_mnist_datasets


class Model(LightningModule):
    def __init__(self) -> None:
        super().__init__()

        self._dataset_dir: str = "./resources"
        self._train_set: Optional[Dataset] = None
        self._val_set: Optional[Dataset] = None
        self._test_set: Optional[Dataset] = None

        self._batch_size: int = 8

        mid_dim: int = 128
        self._module: nn.Module = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1 * 28 * 28, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, 10),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self._module(x)

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adadelta(self.parameters())

    def training_step(self, batch: tuple[Tensor, Tensor], idx: int) -> Tensor:
        loss: Tensor = self._calc_loss(self(batch[0]), batch[1])
        self.log("loss_train", loss)
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], idx: int) -> None:
        loss: Tensor = self._calc_loss(self(batch[0]), batch[1])
        self.log("loss_val", loss)

    def test_step(self, batch: tuple[Tensor, Tensor], idx: int) -> None:
        predict: Tensor = self(batch[0])

    def prepare_data(self) -> None:
        MNIST(self._dataset_dir, train=True, download=True)
        MNIST(self._dataset_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        train_set: Dataset
        val_set: Dataset
        test_set: Dataset
        train_set, val_set, test_set = make_mnist_datasets()

        if stage is None or stage == "fit":
            self._train_set = train_set
            self._val_set = val_set

        if stage is None or stage == "test":
            self._test_set = test_set

    def train_dataloader(self):
        return DataLoader(
            self._train_set,
            batch_size=self._batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(self._val_set, batch_size=self._batch_size, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self._test_set, batch_size=self._batch_size, pin_memory=True)

    def _calc_loss(self, predict: Tensor, label: Tensor) -> Tensor:
        return F.cross_entropy(predict, label)


def train() -> None:
    model: LightningModule = Model()

    total_epochs: int = 50
    trainer: Trainer = Trainer(max_epochs=total_epochs)
    trainer.fit(model)
    trainer.test()
