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

import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from pytorch_lightning import LightningModule, Trainer
from torch import Tensor
from torchvision.datasets import MNIST

from ..data import split_dataset


class Model(LightningModule):
    def __init__(self) -> None:
        super().__init__()

        self._dataset_dir: str = "./resources"
        self._train_set: Optional[Dataset] = None
        self._val_set: Optional[Dataset] = None
        self._test_set: Optional[Dataset] = None

        self._batch_size: int = 120

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
        predict: Tensor = self(batch[0])
        loss: Tensor = self._calc_loss(predict, batch[1])
        self.log("loss/train", loss)
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], idx: int) -> None:
        predict: Tensor = self(batch[0])
        loss: Tensor = self._calc_loss(predict, batch[1])
        acc: Tensor = self._calc_acc(predict, batch[1])
        self.log("loss/val", loss)
        self.log("acc/val", loss)

    def test_step(self, batch: tuple[Tensor, Tensor], idx: int) -> None:
        predict: Tensor = self(batch[0])
        acc: Tensor = self._calc_acc(predict, batch[1])
        self.log("acc/test", acc)

    def prepare_data(self) -> None:
        MNIST(self._dataset_dir, train=True, download=True)
        MNIST(self._dataset_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage is None or stage == "fit":
            train_val_set: Dataset = MNIST(self._dataset_dir, train=True)
            self._train_set, self._val_set = split_dataset(train_val_set, [54000, 6000])

        if stage is None or stage == "test":
            self._test_set = MNIST(self._dataset_dir, train=False)

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

    @staticmethod
    def _calc_loss(predict: Tensor, label: Tensor) -> Tensor:
        return F.cross_entropy(predict, label)

    @staticmethod
    def _calc_acc(predict: Tensor, label: Tensor) -> Tensor:
        return (predict.argmax(-1) == label).sum() / label.shape[0]


def train() -> None:
    model: LightningModule = Model()

    total_epochs: int = 50
    trainer: Trainer = Trainer(max_epochs=total_epochs, gpus=torch.cuda.device_count())
    trainer.fit(model)
    trainer.test()
