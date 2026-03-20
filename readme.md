# WFR-FM Release

## Structure

- `src/`: training and evaluation code
- `data/`: datasets used by the release scripts
- `results/`: packaged checkpoints and run outputs

## Run

Install dependencies first:

```bash
pip install -r requirements.txt
```

Then run a dataset-specific script from `src/`.

Examples:

```bash
cd src

python train_simulation.py
python train_dygen.py
python train_gaussian_1000d.py
python train_emt.py
python train_eb_5d.py
python train_eb_50d.py
python train_eb.py
python train_cite_50d.py
python train_weinreb.py
```

## Outputs

Each script writes to:

```text
results/<experiment_name>/
```

Typical files:

- `pretrain_best_model`
- `evaluation_result.csv`
- `action_result.csv`
- `training_curve.png`
- `sde_point_0.npy`
- `sde_trajec_0.npy`
- `sde_weight_0.npy`

## Config

All experiment settings are defined in:

- `src/experiment_configs.py`

The runner used by all scripts is:

- `src/experiment_runner.py`
