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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from .data import make_mnist_datasets


def train() -> None:
    device: torch.device = torch.device("cpu")

    # make datasets
    train_set: Dataset
    val_set: Dataset
    test_set: Dataset
    train_set, val_set, test_set = make_mnist_datasets()

    # make dataloaders
    batch_size: int = 8
    train_loader: DataLoader = DataLoader(
        train_set, batch_size, shuffle=True, pin_memory=True, drop_last=True
    )
    val_loader: DataLoader = DataLoader(val_set, batch_size, pin_memory=True)
    test_loader: DataLoader = DataLoader(test_set, batch_size, pin_memory=True)

    # model
    model: nn.Module = nn.Sequential(
        nn.Conv2d(1, 8, 3, 1, 1, bias=False),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.Conv2d(8, 16, 3, 1, 1, bias=False),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, 2, 1, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32 * 14 * 14, 512, bias=False),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512, 10),
    ).to(device)

    # optimizer
    optimizer: optim.Optimizer = optim.Adadelta(model.parameters())

    # scheduler
    total_epochs: int = 50
    scheduler: optim.lr_scheduler.LambdaLR = optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: 1 - epoch / total_epochs
    )

    # main loop
    for epoch in range(total_epochs):
        # train
        model.train()
        for data, label in tqdm(
            train_loader, desc=f"Epoch {epoch} | Train", leave=True, unit="step"
        ):
            data: Tensor
            label: Tensor

            data = data.to(device)
            label = label.to(device)

            # calc loss
            predict: Tensor = model(data)
            loss: Tensor = F.cross_entropy(predict, label)

            # update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # validation
        model.eval()
        with torch.no_grad():
            loss_sum: float = 0.0
            for data, label in tqdm(
                val_loader, desc=f"Epoch {epoch} | Validation", leave=True, unit="step"
            ):
                data: Tensor
                label: Tensor

                data = data.to(device)
                label = label.to(device)

                # calc loss
                predict: Tensor = model(data)
                loss: Tensor = F.cross_entropy(predict, label)
                loss_sum += loss.item()

        print(f"Epoch {epoch}:\tValidation Loss = {loss_sum / len(val_loader):.3f}")

        # scheduler
        scheduler.step()

    # test
    model.eval()
    with torch.no_grad():
        correct_num: int = 0
        total_num: int = 0
        for data, label in tqdm(test_loader, desc="Test", leave=True, unit="step"):
            data: Tensor
            label: Tensor

            data = data.to(device)
            label = label.to(device)

            # acc
            predict: Tensor = model(data)
            correct_num += (predict.max(-1)[0] == label).sum().item()
            total_num += label.shape[0]

    print(f"Accuracy: {correct_num / total_num}:.1f")