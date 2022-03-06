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

from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from pytorch_lightning import LightningDataModule
from torchvision.datasets import MNIST

from ..data import make_mnist_transform, split_dataset


class DataModel(LightningDataModule):
    def __init__(self) -> None:
        super().__init__()

        self._dataset_dir: str = "./resources"
        self._train_set: Optional[Dataset] = None
        self._val_set: Optional[Dataset] = None
        self._test_set: Optional[Dataset] = None

        self._batch_size: int = 120

    def prepare_data(self) -> None:
        MNIST(self._dataset_dir, train=True, download=True)
        MNIST(self._dataset_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage is None or stage == "fit":
            train_val_set: Dataset = MNIST(
                self._dataset_dir, train=True, transform=make_mnist_transform()
            )
            self._train_set, self._val_set = split_dataset(train_val_set, [54000, 6000])

        if stage is None or stage == "test":
            self._test_set = MNIST(self._dataset_dir, train=False, transform=make_mnist_transform())

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
