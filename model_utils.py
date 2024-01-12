""" Contains functions for training and testing a nn.model
"""
import os
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Tuple, Dict
from model import EORMulticlassModel
from data_utils import EORDataset, get_train_valid_data
from constants import *


def train_step(model: nn.Module,
               train_dataloader: DataLoader,
               loss_fn: nn.Module,
               optimizer: torch.optim.Optimizer) -> Tuple[float, float]:
    """Performs the training steps of a nn.Module for each epoch

    :param model: a nn.Module
    :param train_dataloader: a DataLoader containing input data
    :param loss_fn: a nn loss function to be used in training
    :param optimizer: a torch.optim.Optimizer to be used in training
    :return: a tuple where the first item is training loss and
            the second items is prediction accuracy of this epoch
    """
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
    """Performs the validation steps of a nn.Module for each epoch

    :param model: a nn.Module
    :param valid_dataloader: a DataLoader containing input data
    :param loss_fn: a nn loss function to be used in training
    :return: a tuple where the first item is validation loss and
            the second items is prediction accuracy of this epoch
    """
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
    """Performs the testing steps of a (trained) nn.Module

    :param model: a nn.Module
    :param test_dataloader: a DataLoader containing test data
    :return: test prediction accuracy
    """
    test_acc = 0
    model.eval()

    with torch.inference_mode():
        for inputs, classes in test_dataloader:
            test_pred_logits = model(inputs)
            calculated_classes = test_pred_logits.argmax(dim=1)
            test_acc += ((calculated_classes == classes).sum().item() / len(calculated_classes))

    test_acc /= len(test_dataloader)

    return test_acc


def train_model(model: nn.Module,
                train_dataset: EORDataset,
                valid_dataset: EORDataset,
                batch_size: int,
                num_epochs: int,
                lr: float) -> Dict[str, float]:
    """Performs the training-validation steps for num_epochs

    :param model: a nn.Module
    :param train_dataset: an EORDataset containing training data
    :param valid_dataset: an EORDataset containing validation data
    :param batch_size: batch size used to create training and validation DataLoaders
    :param num_epochs: number of epochs to train the model
    :param lr: learning rate used in optimizer
    :return: a dictionary containing the following keys and their values:
            train_loss: training loss, train_acc: training prediction accuracy
            valid_loss: validation loss, valid_acc: validation prediction accuracy
    """

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=os.cpu_count())
    valid_dataloader = DataLoader(dataset=valid_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=os.cpu_count())

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

    result_dict = {'train_loss': train_loss,
                   'train_acc': train_acc,
                   'valid_loss': valid_loss,
                   'valid_acc': valid_acc}

    return result_dict


def test_model(model: nn.Module,
               test_dataset: EORDataset,
               batch_size: int) -> float:
    """Test the trained model

    :param model: a nn.Module
    :param test_dataset: an EORDataset containing testing data
    :param batch_size: batch size used to create testing DataLoaders
    :return: the accuracy of test prediction
    """

    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=os.cpu_count())

    test_acc = test_step(model=model, test_dataloader=test_dataloader)

    return test_acc


def save_model(model: nn.Module, params: Dict[str, float]) -> None:
    """Saves a nn.Module to the path defined in the constants.py

    The model name includes hyperparameters received in params.

    :param model: a nn.Module to be saved
    :param params: a dictionary containing the following hyperparameters
            keys: hidden_size, batch_size, learning_rate, num_epochs, train_portion
    :return: None
    """
    # create directory
    model_dir = Path(MODEL_PATH)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_name = f"{MODEL_NAME}_hidden{params['hidden_size']}_batch{params['batch_size']}_lr{params['learning_rate']}" \
                 f"_epochs{params['num_epochs']}_tp{params['train_portion']:.2f}{MODEL_EXTENSION}"
    torch.save(obj=[model.kwargs, model.state_dict()], f=model_dir / model_name)


def run_model(input_dataset: EORDataset,
              test_dataset: EORDataset,
              params: Dict[str, int | float]) -> Dict[str, float]:
    """Creates, trains, tests, and saves an EORMulticlassModel

    :param input_dataset: an EORDataset to be split into training and validation datasets
    :param test_dataset: an EORDataset used for testing the model
    :param params: a dictionary containing the following hyperparameters
            keys: hidden_size, batch_size, learning_rate, num_epochs, train_portion
    :return: a dictionary containing the following keys and their values:
            train_loss: training loss, train_acc: accuracy of training prediction
            valid_loss: validation loss, valid_acc: accuracy of validation prediction
            test_acc: accuracy of testing prediction
    """
    # divide input dataset into training and validation datasets
    train_dataset, valid_dataset = get_train_valid_data(EOR_dataset=input_dataset,
                                                        train_portion=params['train_portion'])
    model = EORMulticlassModel(input_size=INPUT_SIZE,
                               hidden_size=params['hidden_size'],
                               output_size=OUTPUT_SIZE)
    # train the model
    result_dict = train_model(model=model,
                              train_dataset=train_dataset,
                              valid_dataset=valid_dataset,
                              batch_size=params['batch_size'],
                              num_epochs=params['num_epochs'],
                              lr=params['learning_rate'])
    # calculate test results
    test_acc = test_model(model=model,
                          test_dataset=test_dataset,
                          batch_size=params['batch_size'])
    result_dict.update({'test_acc': test_acc})

    # save the model
    save_model(model=model, params=params)

    return result_dict
