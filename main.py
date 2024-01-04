from typing import List, Tuple
from extract_data import read_excel_file
from generate_input import generate_samples, get_train_valid_data, create_dataset, convert_to_PCA, normalize_data
from model_operations import train_model, test_model
from plots import scatter_plot, plot_pca
from test_result_operations import read_results_file, save_results
from constants import *


def run_once(min_list: List[List[str | float]], max_list: List[List[str | float]],
             test_list: List[List[str | float]]) -> Tuple[float, float, float]:
    # create input data
    input_data, input_labels = generate_samples(min_list=min_list, max_list=max_list,
                                                samples_per_class=SAMPLES_PER_CLASS, props_count=NUM_PROPERTIES)
    # transforming data to pca
    if PCA_ENABLED == 1:
        input_data = convert_to_PCA(data=input_data, pca_size=PCA_COMPONENTS)
    # normalize data between 0 and 1
    norm_input_data = normalize_data(input_data=input_data, data=input_data, props_count=NUM_PROPERTIES)
    # plot input using first two Principal Components
    scatter_plot(norm_input_data, input_labels)
    # create input dataset
    input_dataset = create_dataset(data=norm_input_data, labels=input_labels)
    # divide input dataset into training and validation datasets
    train_dataset, valid_dataset = get_train_valid_data(EOR_dataset=input_dataset, train_portion=TRAIN_PORTION)
    # train and save the model
    train_acc, valid_acc = train_model(train_dataset=train_dataset, valid_dataset=valid_dataset,
                                       batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS,
                                       lr=LEARNING_RATE, hidden_size=HIDDEN_SIZE)

    ## testing the model ##
    # get test data
    test_data = [sample[1:] for sample in test_list]
    test_labels = [sample[0] for sample in test_list]
    # transforming data to pca
    if PCA_ENABLED == 1:
        test_data = convert_to_PCA(data=test_data, pca_size=PCA_COMPONENTS)
    # normalize data between 0 and 1
    norm_test_data = normalize_data(input_data=input_data, data=test_data, props_count=NUM_PROPERTIES)
    # create test dataset
    test_dataset = create_dataset(data=norm_test_data, labels=test_labels)
    # calculate test results
    test_acc = test_model(test_dataset=test_dataset, batch_size=BATCH_SIZE, hidden_size=HIDDEN_SIZE)

    return train_acc, valid_acc, test_acc

def run_multiple(min_list: List[List[str | float]], max_list: List[List[str | float]],
                 test_list: List[List[str | float]]):
    # read model parameters to be tested
    param_list, param_labels = read_results_file()
    ## perform the train and test steps once per each set of parameters ##
    for i, params in enumerate(param_list):
        # set parameters
        BATCH_SIZE, HIDDEN_SIZE, LEARNING_RATE, TRAIN_PORTION, SAMPLES_PER_CLASS, NUM_EPOCHS, PCA_ENABLED = params[:-3]
        params[-2:] = run_once(min_list=min_list, max_list=max_list, test_list=test_list)
    # save results to the Excel file
    save_results(data=param_list, labels=param_labels)


if __name__ == "__main__":
    # read oilfield properties
    min_list, max_list, test_list = read_excel_file()
    # run_once(min_list=min_list, max_list=max_list, test_list=test_list)
    run_multiple(min_list=min_list, max_list=max_list, test_list=test_list)
