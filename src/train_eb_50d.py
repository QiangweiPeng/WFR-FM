from experiment_configs import EXPERIMENTS
from experiment_runner import run_experiment


if __name__ == "__main__":
    run_experiment(EXPERIMENTS["eb_50d"])
