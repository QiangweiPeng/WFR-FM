import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import ot
import pandas as pd
import torch
from torchdiffeq import odeint_adjoint as odeint

from models import ODEFunc2
from utils import get_base_model


def _get_model_outputs(f_net, t, data):
    outputs = f_net(t, data)
    if not isinstance(outputs, tuple) or len(outputs) < 2:
        raise ValueError("Model must return at least velocity and growth outputs")
    return outputs[:2]


def load_reducer(reducer_path):
    if reducer_path is None:
        return None
    if not os.path.exists(reducer_path):
        warnings.warn(f"Reducer not found at {reducer_path}; falling back to first two dimensions.")
        return None
    try:
        import joblib
    except ImportError:
        warnings.warn("joblib is not installed; falling back to first two dimensions.")
        return None
    try:
        return joblib.load(reducer_path)
    except Exception as exc:
        warnings.warn(
            f"Failed to load reducer from {reducer_path} ({exc}); "
            "falling back to first two dimensions."
        )
        return None


def evaluate_model(gt_data, model_data, a, b):
    cost = torch.cdist(gt_data, model_data, p=2).cpu().numpy()
    if np.isnan(cost).any() or np.isinf(cost).any():
        return np.nan
    return ot.emd2(a, b, cost, numItermax=int(1e7))


def generate_trajectories_sde(
    df,
    f_net,
    device,
    exp_dir,
    all_times,
    sigma=0.0,
    use_mass=True,
    num_points=None,
    num_runs=1,
    trajectory_bins=100,
):
    del sigma
    time_points = np.array(sorted(all_times), dtype=np.float32)
    start_time = float(time_points[0])
    end_time = float(time_points[-1])
    data = torch.tensor(df[df["samples"] == start_time].values, dtype=torch.float32).requires_grad_()
    x0 = data[:, 1:].to(device).requires_grad_()

    results = []
    for run_idx in range(num_runs):
        if num_points is not None:
            sample_indices = np.random.choice(x0.size(0), size=num_points, replace=False)
            x0_subset = x0[sample_indices].to(device)
        else:
            x0_subset = x0.to(device)

        lnw0 = torch.log(torch.ones(x0_subset.shape[0], 1) / x0_subset.shape[0]).to(device)
        initial_state = (x0_subset, lnw0)

        for param in f_net.parameters():
            param.requires_grad = False

        ts = torch.linspace(start_time, end_time, trajectory_bins, device=device)
        sde_traj, traj_lnw = odeint(ODEFunc2(f_net), initial_state, ts)
        sde_traj = sde_traj.detach().cpu().numpy()

        sample_number = min(100, sde_traj.shape[1])
        sample_indices = np.random.choice(sde_traj.shape[1], size=sample_number, replace=False)
        sampled_sde_traj = sde_traj[:, sample_indices, :]
        np.save(os.path.join(exp_dir, f"sde_trajec_{run_idx}.npy"), sampled_sde_traj)

        ts_points = torch.tensor(time_points, dtype=torch.float32, device=device)
        sde_point, traj_lnw_points = odeint(ODEFunc2(f_net), initial_state, ts_points)
        if use_mass:
            weight = torch.exp(traj_lnw_points)
        else:
            weight = torch.ones_like(traj_lnw_points)

        sde_point_np = sde_point.detach().cpu().numpy()
        weight_np = weight.detach().cpu().numpy()
        np.save(os.path.join(exp_dir, f"sde_point_{run_idx}.npy"), sde_point_np)
        np.save(os.path.join(exp_dir, f"sde_weight_{run_idx}.npy"), weight_np)

        sde_point_tensor = torch.tensor(sde_point_np, dtype=torch.float32)
        sde_weight_tensor = torch.tensor(weight_np, dtype=torch.float32)
        base_count = len(df[df["samples"] == start_time])
        for i in range(1, len(time_points)):
            time_point = time_points[i]
            gt_data = torch.from_numpy(df[df["samples"] == time_point].iloc[:, 1:].values).float()
            a = np.ones((gt_data.shape[0],), dtype=np.float64) / gt_data.shape[0]
            gt_mass = len(df[df["samples"] == time_point]) / base_count
            model_i_data = sde_point_tensor[i].float()
            b = sde_weight_tensor[i].numpy().reshape(-1)
            b = b / b.sum()
            pred_mass = sde_weight_tensor[i].numpy().sum() / sde_weight_tensor[0].numpy().sum()
            w1 = evaluate_model(gt_data, model_i_data, a, b)
            tmv = np.abs(pred_mass - gt_mass) / gt_mass
            results.append(
                {
                    "Time Point": float(time_point),
                    "Model": f"WFR-FM_run{run_idx + 1}",
                    "W1 Distance": w1,
                    "TMV": tmv,
                }
            )

    return results


def aggregate_evaluation_results(results):
    df_results = pd.DataFrame(results)
    if df_results.empty:
        return df_results
    df_results["Base Model"] = df_results["Model"].apply(get_base_model)
    return (
        df_results.groupby(["Time Point", "Base Model"])
        .agg(
            W1_Mean=("W1 Distance", "mean"),
            W1_STD=("W1 Distance", "std"),
            TMV_Mean=("TMV", "mean"),
            TMV_STD=("TMV", "std"),
        )
        .reset_index()
    )


def save_training_curve(losses, v_losses, g_losses, output_file):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(losses, label="total")
    ax.plot(v_losses, label="velocity")
    ax.plot(g_losses, label="growth")
    ax.legend()
    fig.savefig(output_file, bbox_inches="tight")
    plt.close(fig)


def _reduce_array(array, reducer=None):
    array = np.asarray(array, dtype=np.float32)
    if array.shape[-1] <= 2 and reducer is None:
        return array[..., :2]
    flat = array.reshape(-1, array.shape[-1])
    reduced = reducer.transform(flat) if reducer is not None else flat[:, :2]
    return reduced.reshape(*array.shape[:-1], 2)


def _reduce_dataframe(df, feature_cols, reducer=None):
    features = df[feature_cols].to_numpy(dtype=np.float32)
    reduced = reducer.transform(features) if reducer is not None else features[:, :2]
    reduced_df = pd.DataFrame(reduced, columns=["x1", "x2"])
    reduced_df["samples"] = df["samples"].to_numpy()
    return reduced_df


def plot_comparison(df, generated, trajectories, output_file, reducer=None):
    feature_cols = [col for col in df.columns if col != "samples"]
    reduced_df = _reduce_dataframe(df, feature_cols, reducer=reducer)
    generated_2d = _reduce_array(generated, reducer=reducer)
    trajectories_2d = _reduce_array(trajectories, reducer=reducer)

    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    scatter = ax.scatter(
        reduced_df["x1"],
        reduced_df["x2"],
        c=reduced_df["samples"],
        cmap="viridis",
        marker="X",
        alpha=0.3,
        s=60,
    )
    points = generated_2d.reshape(-1, 2)
    time_points = sorted(reduced_df["samples"].unique())
    n_gen = generated_2d.shape[1]
    colors = [state for state in time_points for _ in range(n_gen)]
    ax.scatter(points[:, 0], points[:, 1], c=colors, cmap="viridis", alpha=0.75, s=35)

    for trajectory in np.transpose(trajectories_2d, axes=(1, 0, 2)):
        ax.plot(trajectory[:, 0], trajectory[:, 1], alpha=0.25, color="black")

    ax.set_xlabel("Gene $X_1$")
    ax.set_ylabel("Gene $X_2$")
    fig.colorbar(scatter, ax=ax, label="Time point")
    fig.savefig(output_file, bbox_inches="tight")
    plt.close(fig)


def plot_g_values(
    df,
    f_net,
    device,
    dim,
    output_file,
    reducer=None,
    per_time=False,
    transparent=True,
    report_growth_correlation=False,
):
    time_points = sorted(df["samples"].unique())
    feature_cols = [f"x{i}" for i in range(1, dim + 1)]
    data_by_time = {}

    for time in time_points:
        subset = df[df["samples"] == time]
        data = torch.tensor(subset[feature_cols].values, dtype=torch.float32, device=device)
        with torch.no_grad():
            t = torch.tensor([time], dtype=torch.float32, device=device)
            _, g = _get_model_outputs(f_net, t, data)
        reduced = reducer.transform(subset[feature_cols].values) if reducer is not None else subset[feature_cols].values[:, :2]
        data_by_time[time] = {
            "xy": reduced,
            "g_values": g.detach().cpu().numpy().reshape(-1),
            "raw": subset[feature_cols].values,
        }

    all_g_values = np.concatenate([content["g_values"] for content in data_by_time.values()])
    norm = plt.Normalize(
        vmin=np.percentile(all_g_values, 10),
        vmax=np.percentile(all_g_values, 90),
        clip=True,
    )

    if report_growth_correlation:
        gt_growth = []
        for content in data_by_time.values():
            raw = content["raw"]
            gt_growth.append(raw[:, 1] ** 2 / (1 + raw[:, 1] ** 2))
        gt_growth = np.concatenate(gt_growth)
        corr = np.corrcoef(all_g_values.flatten(), gt_growth.flatten())[0, 1]
        print(f"growth correlation: {corr:.4f}")

    if per_time:
        for time, content in data_by_time.items():
            fig, ax = plt.subplots(figsize=(12, 8))
            colors = plt.cm.plasma(norm(content["g_values"]))
            ax.scatter(content["xy"][:, 0], content["xy"][:, 1], color=colors, alpha=0.7, marker="o")
            ax.set_xlabel("Gene $X_1$")
            ax.set_ylabel("Gene $X_2$")
            sm = plt.cm.ScalarMappable(cmap="plasma", norm=norm)
            sm.set_array(all_g_values)
            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label("Predicted growth rate")
            target = output_file.replace(".png", f"_{time}.png")
            fig.savefig(target, bbox_inches="tight", transparent=transparent)
            plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    for time, content in data_by_time.items():
        colors = plt.cm.plasma(norm(content["g_values"]))
        ax.scatter(content["xy"][:, 0], content["xy"][:, 1], color=colors, alpha=0.7, marker="o", label=f"Time {time}")
    ax.set_xlabel("Gene $X_1$")
    ax.set_ylabel("Gene $X_2$")
    sm = plt.cm.ScalarMappable(cmap="plasma", norm=norm)
    sm.set_array(all_g_values)
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Predicted growth rate")
    fig.savefig(output_file, bbox_inches="tight", transparent=transparent)
    plt.close(fig)
