import os
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from generate_input import EORDataset
from typing import Tuple
from constants import OUTPUT_SIZE, INPUT_SIZE, MODEL_PATH, MODEL_NAME
from model import EORMulticlassModel


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
    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)

    return train_loss, train_acc


def valid_step(model: nn.Module,
               valid_dataloader: torch.utils.data.DataLoader,
               loss_fn: nn.Module) -> Tuple[float, float]:

    valid_loss, valid_acc = 0, 0
    model.eval()

    with torch.inference_mode():
        for inputs, classes in valid_dataloader:
            valid_pred_logits = model(inputs)
            loss = loss_fn(valid_pred_logits, classes)
            valid_loss += loss.item()

            outputs_class = valid_pred_logits.argmax(dim=1)
            valid_acc += ((outputs_class == classes).sum().item() / len(outputs_class))

    # Adjust metrics to get average loss and accuracy per batch
    valid_loss /= len(valid_dataloader)
    valid_acc /= len(valid_dataloader)

    return valid_loss, valid_acc


def test_step(model: nn.Module, test_dataloader: torch.utils.data.DataLoader) -> float:

    test_acc = 0
    model.eval()

    with torch.inference_mode():
        for inputs, classes in test_dataloader:
            test_pred_logits = model(inputs)
            calculated_classes = test_pred_logits.argmax(dim=1)
            test_acc += ((calculated_classes == classes).sum().item() / len(calculated_classes))

    test_acc /= len(test_dataloader)

    return test_acc


def train_model(train_dataset: EORDataset, valid_dataset: EORDataset,
                batch_size: int, num_epochs: int, lr: float,
                hidden_size: int, input_size: int = INPUT_SIZE,
                output_size: int = OUTPUT_SIZE, loss_function: str = "",
                optimization_alg: str = "") -> Tuple[float, float]:

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=os.cpu_count())
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False,
                                  num_workers=os.cpu_count())
    model = EORMulticlassModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        train_loss, train_acc = train_step(model=model,
                                           train_dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)

        valid_loss, valid_acc = valid_step(model=model,
                                           valid_dataloader=valid_dataloader,
                                           loss_fn=loss_fn)

        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.3f} | "
            f"train_acc: {train_acc:.3f} | "
            f"valid_loss: {valid_loss:.3f} | "
            f"valid_acc: {valid_acc:.3f}"
        )

    # save the model
    model_dir = Path(MODEL_PATH)
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(obj=model.state_dict(), f=model_dir/MODEL_NAME)

    return train_acc, valid_acc


def test_model(test_dataset: EORDataset, batch_size: int, hidden_size: int,
               input_size: int = INPUT_SIZE,  output_size: int = OUTPUT_SIZE) -> float:

    model = EORMulticlassModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    model.load_state_dict(torch.load(f=f"{MODEL_PATH}/{MODEL_NAME}"))

    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count())

    test_acc = test_step(model=model, test_dataloader=test_dataloader)
    print(f"test_acc: {test_acc}")

    return test_acc
