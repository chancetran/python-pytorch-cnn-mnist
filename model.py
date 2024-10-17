"""
Module defining Model torch.nn.Module class.

Source: https://github.com/pytorch/examples/blob/main/mnist/main.py
"""

import torch


class Model(torch.nn.Module):
    def __init__(self) -> None:

        super(Model, self).__init__()

        self.conv_1 = torch.nn.Conv2d(1, 32, 3, 1)
        self.conv_2 = torch.nn.Conv2d(32, 64, 3, 1)
        self.dropout_1 = torch.nn.Dropout(0.25)
        self.fc_1 = torch.nn.Linear(9216, 128)
        self.dropout_2 = torch.nn.Dropout(0.5)
        self.fc_2 = torch.nn.Linear(128, 10)

    def forward(self, x: torch.tensor) -> float:

        x = self.conv_1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv_2(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.dropout_1(x)
        x = torch.flatten(x, 1)
        x = self.fc_1(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout_2(x)
        x = self.fc_2(x)

        output = torch.nn.functional.log_softmax(x, dim=1)

        return output
