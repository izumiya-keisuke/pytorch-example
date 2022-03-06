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

import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader


class MyDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()

        self._data_num: int = 80
        data_size: tuple[int, ...] = (10, 20)

        self._data: np.ndarray = np.random.rand(self._data_num, *data_size)

    def __len__(self) -> int:
        return self._data_num

    def __getitem__(self, idx) -> np.ndarray:
        return self._data[idx]


def make_dataloader() -> DataLoader:
    return DataLoader(
        MyDataset(), batch_size=4, shuffle=True, pin_memory=True, drop_last=True
    )
