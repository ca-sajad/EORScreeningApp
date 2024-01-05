import optuna
from extract_data import read_excel_file
from manual_opt import run_once


def objective(trial):
    hyper_params = {
        'hidden_size': trial.suggest_int(name='hidden_size', low=16, high=128, step=16),
        'learning_rate': trial.suggest_float(name='learning_rate', low=0.01, high=0.15, step=0.02),
        'pca_enabled': trial.suggest_categorical('pca_enabled', [0, 1]),
        'batch_size': trial.suggest_int(name='batch_size', low=16, high=128, step=16),
        'num_epochs': trial.suggest_int(name='num_epochs', low=100, high=300, step=50),
        'samples_per_class': trial.suggest_int(name='samples_per_class', low=100, high=200, step=50),
        'train_portion': trial.suggest_float(name='train_portion', low=0.7, high=0.9, step=0.05)
    }

    min_list, max_list, test_list = read_excel_file()
    train_acc, valid_acc, test_acc = run_once(min_list=min_list, max_list=max_list,
                                              test_list=test_list, params=hyper_params)

    return test_acc


def find_optimum_model():
    study = optuna.create_study(direction='maximize', study_name="EOR_ANN_model")
    study.optimize(objective, n_trials=100)

    # Train the final model with the best hyperparameters
    min_list, max_list, test_list = read_excel_file()
    print(f"\n\nBest Model Results:")
    run_once(min_list=min_list, max_list=max_list, test_list=test_list, params=study.best_params)


if __name__ == "__main__":
    find_optimum_model()
