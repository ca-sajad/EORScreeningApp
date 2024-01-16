
import torch
from torch import nn


class EORMulticlassModel(nn.Module):
    """a nn.Module model
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        """Initializes an EORMulticlassModel

        :param input_size: Size of the input features.
        :param hidden_size: Size of the hidden layer.
        :param output_size: Size of the output layer.
        """
        super().__init__()
        self.kwargs = {'input_size': input_size, 'hidden_size': hidden_size, 'output_size': output_size}
        self.block_1 = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=output_size)
        )

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model

        :param input_data: a torch.Tensor as the input data of the model
        :return: a torch.Tensor
        """
        return self.block_1(input_data)
