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

from typing import Union

import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torch import Tensor

from .data import DataModel


class Model(LightningModule):
    def __init__(self) -> None:
        super().__init__()

        self._module: nn.Module = self._make_model()

    def forward(self, x: Tensor) -> Tensor:
        return self._module(x)

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adadelta(self.parameters())

    def training_step(self, batch: tuple[Tensor, Tensor], idx: int) -> Tensor:
        predict: Tensor = self(batch[0])
        loss: Tensor = self._calc_loss(predict, batch[1])
        return loss

    def training_epoch_end(self, outputs: list[Tensor]) -> None:
        self.log("loss/train", sum(outputs) / len(outputs))

    def validation_step(self, batch: tuple[Tensor, Tensor], idx: int) -> dict[str, Tensor]:
        predict: Tensor = self(batch[0])
        loss: Tensor = self._calc_loss(predict, batch[1])
        acc: Tensor = self._calc_acc(predict, batch[1])
        return {"loss": loss, "acc": acc}

    def validation_epoch_end(self, outputs: list[dict[str, Tensor]]) -> None:
        loss_sum: Union[float, Tensor] = 0.0
        acc_sum: Union[float, Tensor] = 0.0
        for output in outputs:
            output: dict[str, Tensor]

            loss_sum += output["loss"]
            acc_sum += output["acc"]

        self.log("loss/val", loss_sum / len(outputs))
        self.log("acc/val", acc_sum / len(outputs))

    def test_step(self, batch: tuple[Tensor, Tensor], idx: int) -> Tensor:
        predict: Tensor = self(batch[0])
        acc: Tensor = self._calc_acc(predict, batch[1])
        return acc

    def test_epoch_end(self, outputs: list[Tensor]) -> None:
        self.log("acc/test", sum(outputs) / len(outputs))

    @staticmethod
    def _make_model() -> nn.Module:
        mid_dim: int = 128
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1 * 28 * 28, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, 10),
        )

    @staticmethod
    def _calc_loss(predict: Tensor, label: Tensor) -> Tensor:
        return F.cross_entropy(predict, label)

    @staticmethod
    def _calc_acc(predict: Tensor, label: Tensor) -> Tensor:
        return (predict.argmax(-1) == label).sum() / label.shape[0]


def train() -> None:
    data_model: LightningDataModule = DataModel()
    model: LightningModule = Model()

    total_epochs: int = 50
    trainer: Trainer = Trainer(max_epochs=total_epochs, gpus=torch.cuda.device_count())
    trainer.fit(model, data_model)
    trainer.test()
