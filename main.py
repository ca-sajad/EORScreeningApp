from extract_data import extract
from generate_input import generate_samples, get_train_valid_data, create_dataset, convert_to_PCA, normalize_data
from model import train_model
from plots import scatter_plot


NUM_SAMPLES = 100   # number of samples per class

def main():
    ave_list, std_dev_list, min_list, max_list = extract()
    input_data, input_labels = generate_samples(ave_list, std_dev_list, min_list, max_list, NUM_SAMPLES)
    input_data = normalize_data(input_data)
    scatter_plot(input_data, input_labels)
    # input_data = convert_to_PCA(input_data)
    input_dataset = create_dataset(input_data, input_labels)
    train_dataset, valid_dataset = get_train_valid_data(input_dataset)
    train_model(train_dataset, valid_dataset)




if __name__ == "__main__":
    main()
