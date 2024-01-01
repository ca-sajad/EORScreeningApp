import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from generate_input import EORDataset
from typing import Tuple

BATCH_SIZE = 32
INPUT_SIZE = 7
HIDDEN_SIZE = 32
OUTPUT_SIZE = 2
NUM_EPOCHS = 1
LEARNING_RATE = 0.01


class EORMulticlassModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=output_size)
        )

    def forward(self, x: torch.Tensor):
        return self.block_1(x)


def train_step(model: nn.Module,
               train_dataloader: DataLoader,
               loss_fn: nn.Module,
               optimizer: torch.optim.Optimizer) -> Tuple[float, float]:
    model.train()
    train_loss, train_acc = 0, 0

    for inputs, classes in train_dataloader:
        outputs = model(inputs)

        loss = loss_fn(outputs, classes)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        outputs_class = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
        train_acc += (outputs_class == classes).sum().item() / len(outputs_class)

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(train_dataloader)
    train_acc = train_acc / len(train_dataloader)

    return train_loss, train_acc


def valid_step(model: nn.Module,
               valid_dataloader: torch.utils.data.DataLoader,
               loss_fn: nn.Module) -> Tuple[float, float]:

    valid_loss, valid_acc = 0, 0
    model.eval()

    with torch.inference_mode():
        for inputs, classes in valid_dataloader:
            test_pred_logits = model(inputs)
            loss = loss_fn(test_pred_logits, classes)
            valid_loss += loss.item()

            outputs_class = test_pred_logits.argmax(dim=1)
            valid_acc += ((outputs_class == classes).sum().item() / len(outputs_class))

    # Adjust metrics to get average loss and accuracy per batch
    valid_loss = valid_loss / len(valid_dataloader)
    valid_acc = valid_acc / len(valid_dataloader)

    return valid_loss, valid_acc


def train_model(train_dataset: EORDataset, valid_dataset: EORDataset):

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=os.cpu_count())
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=os.cpu_count())
    model = EORMulticlassModel(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_step(model=model,
                                           train_dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)

        # valid_loss, valid_acc = valid_step(model=model,
        #                                    valid_dataloader=valid_dataloader,
        #                                    loss_fn=loss_fn)
        valid_loss, valid_acc = 0, 0

        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"valid_loss: {valid_loss:.4f} | "
            f"valid_acc: {valid_acc:.4f}"
        )

