import os
import random

import numpy as np
import pandas as pd
import torch

from ema import EMA
from evaluation import (
    aggregate_evaluation_results,
    generate_trajectories_sde,
    load_reducer,
    plot_comparison,
    plot_g_values,
    save_training_curve,
)
from models import FNet
from train import pretrain
from utils import compute_trajectories_action


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def set_seeds(seed):
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_repo_path(path):
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def compute_relative_mass(df):
    sample_sizes = df.groupby("samples").size()
    ref0 = sample_sizes / sample_sizes.iloc[0]
    return torch.tensor(ref0.values, dtype=torch.float32)


def run_experiment(config):
    set_seeds(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = resolve_repo_path(config.data_file)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    output_dir = resolve_repo_path(os.path.join("results", config.experiment_name))
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    df = df.iloc[:, : config.dim + 1]

    model = FNet(
        in_out_dim=config.dim,
        hidden_dim=config.hidden_dim,
        n_hiddens=config.n_hiddens,
        activation="leakyrelu",
    ).to(device)

    optimizer_1 = torch.optim.Adam(model.v_net.parameters(), lr=config.lr_v)
    optimizer_2 = torch.optim.Adam(model.g_net.parameters(), lr=config.lr_g)
    scheduler_1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_1, T_max=config.n_epoch, eta_min=config.eta_min)
    scheduler_2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_2, T_max=config.n_epoch, eta_min=config.eta_min)
    relative_mass = compute_relative_mass(df)
    ema = EMA(model, decay=config.ema_decay) if config.ema_decay is not None else None

    model, v_losses, g_losses, losses = pretrain(
        model,
        df,
        optimizer_1,
        optimizer_2,
        scheduler_1=scheduler_1,
        scheduler_2=scheduler_2,
        delta=config.delta,
        batch_size=config.batch_size,
        n_epoch=config.n_epoch,
        hold_out=config.hold_out,
        relative_mass=relative_mass,
        use_mini_batch=config.use_mini_batch,
        chunk_size=config.chunk_size,
        ema=ema,
        device=device,
    )

    save_training_curve(losses, v_losses, g_losses, os.path.join(output_dir, "training_curve.png"))
    torch.save(model.state_dict(), os.path.join(output_dir, "pretrain_best_model"))

    if ema is not None and config.apply_ema_for_eval:
        ema.apply_shadow()

    reducer = load_reducer(resolve_repo_path(config.reducer_path))

    plot_g_values(
        df,
        model,
        device=device,
        dim=config.dim,
        output_file=os.path.join(output_dir, "gene_growth_pre_post.png"),
        reducer=reducer,
        per_time=config.plot_growth_per_time,
        transparent=config.plot_transparent,
        report_growth_correlation=config.report_growth_correlation,
    )

    groups = sorted(df.samples.unique())
    results = generate_trajectories_sde(df, model, device, output_dir, groups, 0.0, use_mass=True, num_points=None, num_runs=1)
    aggregated = aggregate_evaluation_results(results)
    aggregated.to_csv(os.path.join(output_dir, "evaluation_result.csv"), index=False)

    action_results = pd.DataFrame(
        compute_trajectories_action(df, model, device, groups, 0.0, use_mass=True, delta=config.delta)
    )
    action_results.to_csv(os.path.join(output_dir, "action_result.csv"), index=False)

    if config.plot_comparison:
        generated = np.load(os.path.join(output_dir, "sde_point_0.npy"), allow_pickle=True)
        trajectories = np.load(os.path.join(output_dir, "sde_trajec_0.npy"), allow_pickle=True)
        plot_comparison(df, generated, trajectories, os.path.join(output_dir, "comparision.png"), reducer=reducer)

    if ema is not None and config.apply_ema_for_eval:
        ema.restore()

    return {
        "output_dir": output_dir,
        "model_path": os.path.join(output_dir, "pretrain_best_model"),
        "metrics_path": os.path.join(output_dir, "evaluation_result.csv"),
    }
