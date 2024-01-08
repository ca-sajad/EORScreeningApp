import optuna
from model_utils import run_model, load_data


def objective(trial) -> float:
    hyper_params = {
        'hidden_size': trial.suggest_int(name='hidden_size', low=32, high=128, step=16),
        'batch_size': trial.suggest_int(name='batch_size', low=32, high=128, step=16),
        'learning_rate': trial.suggest_float(name='learning_rate', low=0.05, high=0.15, step=0.02),
        'num_epochs': trial.suggest_int(name='num_epochs', low=150, high=300, step=50),
        'train_portion': trial.suggest_float(name='train_portion', low=0.7, high=0.95, step=0.05),
    }

    input_dataset, test_dataset = load_data()
    result_dict = run_model(input_dataset=input_dataset,
                            test_dataset=test_dataset,
                            params=hyper_params)

    return result_dict['test_acc']


def find_optimum_model() -> None:
    study = optuna.create_study(direction='maximize', study_name="EOR_ANN_model")
    study.optimize(objective, n_trials=50)
    best_params = study.best_params
    print(f"\nBest Model Parameters:")
    for key, value in best_params.items():
        print(f"parameter: {key}, value: {value}")
    print(f"best accuracy: {study.best_value}")


if __name__ == "__main__":
    find_optimum_model()
