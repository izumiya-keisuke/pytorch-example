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

from typing import Callable, Optional, Sequence

import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import MNIST


class MyDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()

        self._data_num: int = 80
        data_size: tuple[int, ...] = (10, 20)

        self._data: np.ndarray = np.random.rand(self._data_num, *data_size)

        self._transform: Optional[Callable[[np.ndarray], np.ndarray]] = None

    def __len__(self) -> int:
        return self._data_num

    def __getitem__(self, idx) -> np.ndarray:
        if self._transform is None:
            return self._data[idx]

        return self._transform(self._data[idx])


def split_dataset(dataset: Dataset, lengths: Sequence[int]) -> list[Dataset]:
    return torch.utils.data.random_split(dataset, lengths)


def make_mnist_dataset() -> tuple[Dataset, Dataset, Dataset]:
    transform: Callable = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    root: str = "./resources"

    train_val_set: MNIST = MNIST(root, train=True, transform=transform, download=True)
    train_len: int = int(len(train_val_set) * 0.8)
    val_len: int = len(train_val_set) - train_len
    train_set: Dataset
    val_set: Dataset
    train_set, val_set = split_dataset(train_val_set, [train_len, val_len])

    test_set: Dataset = MNIST(root, train=False, transform=transform, download=True)

    return test_set, val_set, test_set


def make_dataloader(dataset: Dataset) -> DataLoader:
    return DataLoader(dataset, batch_size=4, shuffle=True, pin_memory=True, drop_last=True)
