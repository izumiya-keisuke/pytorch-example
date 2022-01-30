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
