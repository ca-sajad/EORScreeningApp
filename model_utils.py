import os
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Tuple, Dict
from model import EORMulticlassModel
from extract_data import read_distribution_file, read_test_dataset_file
from data_utils import EORDataset, generate_samples, calculate_minmax, \
    get_train_valid_data, create_dataset, normalize_data, save_mins_maxs
from plots import scatter_plot
from constants import *


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
                hidden_size: int, input_size: int,
                output_size: int = OUTPUT_SIZE) -> Dict[str, float]:
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=os.cpu_count())
    valid_dataloader = DataLoader(dataset=valid_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=os.cpu_count())
    model = EORMulticlassModel(input_size=input_size,
                               hidden_size=hidden_size,
                               output_size=output_size)
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
    torch.save(obj=model.state_dict(), f=model_dir / MODEL_NAME)

    result_dict = {'train_loss': train_loss,
                   'train_acc': train_acc,
                   'valid_loss': valid_loss,
                   'valid_acc': valid_acc}

    return result_dict


def test_model(test_dataset: EORDataset, batch_size: int, hidden_size: int,
               input_size: int, output_size: int = OUTPUT_SIZE) -> float:
    model = EORMulticlassModel(input_size=input_size,
                               hidden_size=hidden_size,
                               output_size=output_size)
    model.load_state_dict(torch.load(f=f"{MODEL_PATH}/{MODEL_NAME}"))

    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=os.cpu_count())

    test_acc = test_step(model=model, test_dataloader=test_dataloader)

    return test_acc


def load_data() -> Tuple[EORDataset, EORDataset]:
    # read minimum and maximum of EOR properties,
    # input_data_dict is in this form: {'min_list':list, 'max_list':list, 'prop_labels':list}
    input_data_dict = read_distribution_file()
    # create input data
    input_data, input_labels = generate_samples(data_dict=input_data_dict,
                                                samples_per_class=SAMPLES_PER_CLASS,
                                                props_count=INPUT_SIZE)
    # calculate minimum and maximum of generated input_data
    min_properties, max_properties = calculate_minmax(input_data)
    # save mins and maxs to a json file
    save_mins_maxs(mins=min_properties,
                   maxs=max_properties,
                   labels=input_data_dict['prop_labels'],
                   file=MIN_MAX_FILE)
    # normalize data between 0 and 1
    norm_input_data = normalize_data(mins=min_properties,
                                     maxs=max_properties,
                                     data=input_data)
    # plot input using first two Principal Components
    # scatter_plot(norm_input_data, input_labels)
    # create input dataset
    input_dataset = create_dataset(data=norm_input_data, labels=input_labels)

    # get test data
    test_data, test_labels = read_test_dataset_file()
    # normalize data between 0 and 1
    norm_test_data = normalize_data(mins=min_properties,
                                    maxs=max_properties,
                                    data=test_data)
    # create test dataset
    test_dataset = create_dataset(data=norm_test_data, labels=test_labels)

    return input_dataset, test_dataset


def run_model(input_dataset: EORDataset, test_dataset: EORDataset,
              params: Dict[str, int | float]) -> Dict[str, float]:
    # divide input dataset into training and validation datasets
    train_dataset, valid_dataset = get_train_valid_data(EOR_dataset=input_dataset,
                                                        train_portion=params['train_portion'])
    # train and save the model
    result_dict = train_model(train_dataset=train_dataset, valid_dataset=valid_dataset,
                              batch_size=params['batch_size'],
                              num_epochs=params['num_epochs'],
                              lr=params['learning_rate'],
                              hidden_size=params['hidden_size'],
                              input_size=INPUT_SIZE)

    # calculate test results
    test_acc = test_model(test_dataset=test_dataset,
                          batch_size=params['batch_size'],
                          hidden_size=params['hidden_size'],
                          input_size=INPUT_SIZE)

    result_dict.update({'test_acc': test_acc})

    return result_dict
