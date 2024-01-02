from extract_data import read_excel_file
from generate_input import generate_samples, get_train_valid_data, create_dataset, convert_to_PCA, normalize_data
from model_operations import train_model, test_model
from plots import scatter_plot
from constants import *


def main():
    ave_list, std_dev_list, min_list, max_list, test_list = read_excel_file()
    # create input dataset
    input_data, input_labels = generate_samples(ave_list=ave_list, std_dev_list=std_dev_list, min_list=min_list,
                                                max_list=max_list, samples_per_class=SAMPLES_PER_CLASS,
                                                props_count=NUM_PROPERTIES)
    input_data = normalize_data(input_data=input_data, data=input_data, props_count=NUM_PROPERTIES)
    scatter_plot(input_data, input_labels)
    #### input_data = convert_to_PCA(data=input_data, pca_size=PCA_COMPONENTS)
    input_dataset = create_dataset(data=input_data, labels=input_labels)
    # divide input dataset into training and validation datasets
    train_dataset, valid_dataset = get_train_valid_data(EOR_dataset=input_dataset, train_portion=TRAIN_PORTION)
    # train and save the model
    train_model(train_dataset=train_dataset, valid_dataset=valid_dataset, batch_size=BATCH_SIZE,
                num_epochs=NUM_EPOCHS, lr=LEARNING_RATE, hidden_size=HIDDEN_SIZE)
    # test the model
    test_data = [sample[1:] for sample in test_list]
    test_labels = [sample[0] for sample in test_list]
    test_data = normalize_data(input_data=input_data, data=test_data, props_count=NUM_PROPERTIES)
    test_dataset = create_dataset(data=test_data, labels=test_labels)
    test_model(test_dataset=test_dataset, batch_size=BATCH_SIZE, hidden_size=HIDDEN_SIZE)


if __name__ == "__main__":
    main()
