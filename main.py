from extract_data import extract
from generate_input import generate_samples, get_train_valid_data, create_dataset
from model import train_model


NUM_SAMPLES = 100

def main():
    ave_list, std_dev_list = extract()
    input_data, input_labels = generate_samples(ave_list, std_dev_list, NUM_SAMPLES)
    input_dataset = create_dataset(input_data, input_labels)
    train_dataset, valid_dataset = get_train_valid_data(input_dataset)
    train_model(train_dataset, valid_dataset)


if __name__ == "__main__":
    main()
