import os
import random
import re

import numpy as np
import ot
import torch
from torchdiffeq import odeint_adjoint as odeint
from tqdm import tqdm

from models import ODEFunc2


def to_np(data):
    return data.detach().cpu().numpy()


def generate_steps(groups):
    return list(zip(groups[:-1], groups[1:]))


def get_base_model(model_name):
    return re.sub(r"_run\d+$", "", model_name)


def compute_uot_plans(X, t_train, delta=1, use_mini_batch_uot=False, chunk_size=1000, draw=False):
    gamma0_plans = []
    gamma1_plans = []
    sampling_info_plans = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_action = 0

    for i in tqdm(range(len(t_train) - 1), desc="Computing UOT plans..."):
        x_source, x_target = X[i], X[i + 1]
        n_source, n_target = x_source.shape[0], x_target.shape[0]
        a = np.ones(n_source)
        b = np.ones(n_target)

        norm_2_dist = ot.dist(x_source, x_target, metric="euclidean")
        cos_sq = np.cos(np.minimum(norm_2_dist / (2 * delta), np.pi / 2)) ** 2
        cost_matrix = -np.log(np.where(cos_sq == 0, 1e-10, cos_sq))

        if not use_mini_batch_uot:
            a_cuda = torch.from_numpy(a).to(device)
            b_cuda = torch.from_numpy(b).to(device)
            cost_matrix_cuda = torch.from_numpy(cost_matrix).to(device)
            G = ot.unbalanced.mm_unbalanced(a_cuda, b_cuda, cost_matrix_cuda, reg_m=[1.0, 1.0])
            total_action += 2 * (delta**2) * ot.unbalanced.mm_unbalanced2(
                a_cuda, b_cuda, cost_matrix_cuda, reg_m=[1.0, 1.0], returnCost="total"
            )
            G = G.cpu().numpy()
            sampling_info_plans.append(None)
        else:
            group_number = n_source // chunk_size + 1
            G = np.zeros((n_source, n_target))
            source_perm = np.arange(n_source)
            np.random.shuffle(source_perm)
            target_perm = np.arange(n_target)
            np.random.shuffle(target_perm)
            source_indices_groups = np.array_split(source_perm, group_number)
            target_indices_groups = np.array_split(target_perm, group_number)

            gamma0_sub_plans = []
            for src_idx, tgt_idx in zip(source_indices_groups, target_indices_groups):
                sub_cost_matrix = cost_matrix[np.ix_(src_idx, tgt_idx)]
                sub_a = a[src_idx]
                sub_b = b[tgt_idx]
                sub_a_cuda = torch.from_numpy(sub_a).to(device)
                sub_b_cuda = torch.from_numpy(sub_b).to(device)
                sub_cost_matrix = torch.from_numpy(sub_cost_matrix).to(device)
                G_sub = ot.unbalanced.mm_unbalanced(sub_a_cuda, sub_b_cuda, sub_cost_matrix, reg_m=[1.0, 1.0])
                G_sub = G_sub.cpu().numpy()
                G[np.ix_(src_idx, tgt_idx)] = G_sub

                g_sub_sum_1 = G_sub.sum(1)
                gamma0_sub = ((sub_a / (g_sub_sum_1 + 1e-12))[:, None]) * G_sub
                gamma0_sub_plans.append(gamma0_sub.astype(np.float32))

            sampling_info = {
                "sub_plans": gamma0_sub_plans,
                "source_groups": source_indices_groups,
                "target_groups": target_indices_groups,
            }
            sampling_info_plans.append(sampling_info)

        g_sum_1 = G.sum(1)
        g_sum_0 = G.sum(0)
        gamma0_plan = ((a / (g_sum_1 + 1e-12))[:, None]) * G
        gamma1_plan = (b / (g_sum_0 + 1e-12)) * G

        gamma0_plans.append(gamma0_plan)
        gamma1_plans.append(gamma1_plan)

    print(f"total_action: {total_action / X[0].shape[0]}")
    return gamma0_plans, gamma1_plans, sampling_info_plans


def sample_from_ot_plan(ot_plan, x0, x1, batch_size, sampling_info=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if sampling_info is None:
        pi_cuda = torch.from_numpy(ot_plan.astype(np.float32)).to(device)
        row_sums = pi_cuda.sum(axis=1)
        total_sum = row_sums.sum()
        if total_sum < 1e-9:
            return np.array([]), np.array([]), np.array([]), np.array([])
        row_probs = row_sums / total_sum
        i_samples = torch.multinomial(row_probs, num_samples=batch_size, replacement=True)
        selected_rows = pi_cuda[i_samples]
        selected_row_sums = row_sums[i_samples]
        conditional_probs = selected_rows / (selected_row_sums.unsqueeze(1) + 1e-12)
        j_samples = torch.multinomial(conditional_probs, num_samples=1).squeeze(1)
        i, j = i_samples.cpu().numpy(), j_samples.cpu().numpy()
    else:
        g_subs = sampling_info["sub_plans"]
        source_indices_groups = sampling_info["source_groups"]
        target_indices_groups = sampling_info["target_groups"]
        block_masses = [g.sum() for g in g_subs]
        total_mass = sum(block_masses)
        if total_mass < 1e-9:
            return np.array([]), np.array([]), np.array([]), np.array([])
        block_probs = torch.tensor(block_masses, dtype=torch.float32, device=device) / total_mass
        sampled_group_indices = torch.multinomial(block_probs, num_samples=batch_size, replacement=True)
        g_subs_gpu = [torch.from_numpy(g).to(device) for g in g_subs]
        source_indices_gpu = [torch.from_numpy(idx).to(device) for idx in source_indices_groups]
        target_indices_gpu = [torch.from_numpy(idx).to(device) for idx in target_indices_groups]
        unique_groups, counts = torch.unique(sampled_group_indices, return_counts=True)
        final_i_samples = torch.empty(batch_size, dtype=torch.int64, device=device)
        final_j_samples = torch.empty(batch_size, dtype=torch.int64, device=device)
        for group_idx, count in zip(unique_groups, counts):
            g_sub = g_subs_gpu[group_idx]
            sub_row_sums = g_sub.sum(axis=1)
            if sub_row_sums.sum() < 1e-9:
                continue
            sub_row_probs = sub_row_sums / sub_row_sums.sum()
            i_local = torch.multinomial(sub_row_probs, num_samples=count.item(), replacement=True)
            selected_sub_rows = g_sub[i_local]
            selected_sub_row_sums = sub_row_sums[i_local]
            sub_cond_probs = selected_sub_rows / (selected_sub_row_sums.unsqueeze(1) + 1e-12)
            j_local = torch.multinomial(sub_cond_probs, num_samples=1).squeeze(1)
            global_i = source_indices_gpu[group_idx][i_local]
            global_j = target_indices_gpu[group_idx][j_local]
            mask = sampled_group_indices == group_idx
            final_i_samples[mask] = global_i
            final_j_samples[mask] = global_j
        i, j = final_i_samples.cpu().numpy(), final_j_samples.cpu().numpy()

    if i.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    return x0[i], x1[j], i, j


def compute_xt_ut_gt(t_relative, delta_t, x0, x1, mass0, mass1, delta):
    index = torch.norm(x1 - x0, dim=1) < torch.pi * delta

    t_relative = t_relative[index]
    x0 = x0[index]
    x1 = x1[index]
    mass0 = mass0[index]
    mass1 = mass1[index]

    diff = x1 - x0
    norm = torch.norm(diff, dim=1, keepdim=True)
    norm_vector = diff / (norm + 1e-9)

    tau = torch.tan(norm / (2 * delta))
    scale = torch.sqrt(mass0 * mass1 / (1 + tau**2))
    omega = 2 * delta * tau * scale
    omega_vector = omega * norm_vector

    A = mass1 + mass0 - 2 * scale
    B = mass0 - scale
    inv_sqrt_am0_m_bsq = 2 * delta / (omega + 1e-9)

    xt_samp = x0 + omega_vector * (
        inv_sqrt_am0_m_bsq
        * (
            torch.arctan((A * t_relative - B) * inv_sqrt_am0_m_bsq)
            - torch.arctan(-B * inv_sqrt_am0_m_bsq)
        )
    )

    masst_samp = A * t_relative**2 - 2 * B * t_relative + mass0
    dmasst_dt = 2 * A * t_relative - 2 * B
    gt_samp = dmasst_dt / masst_samp * (1 / delta_t)
    ut_samp = omega_vector / masst_samp * (1 / delta_t)

    return xt_samp, gt_samp, ut_samp, masst_samp / mass0, index


def get_batch(X, t_train, batch_size, gamma0_plans, gamma1_plans, delta, ratios, sampling_info_plans):
    ts = []
    xts = []
    uts = []
    gts = []
    massts = []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for t in range(len(t_train) - 1):
        gamma0_plan = gamma0_plans[t]
        gamma1_plan = gamma1_plans[t]
        sampling_info = sampling_info_plans[t]

        x0, x1, idx_0, idx_1 = sample_from_ot_plan(gamma0_plan, X[t], X[t + 1], batch_size, sampling_info)
        x0 = torch.from_numpy(x0).float().to(device)
        x1 = torch.from_numpy(x1).float().to(device)

        mass0 = torch.from_numpy(gamma0_plan[idx_0, idx_1].reshape(-1, 1)).float().to(device)
        mass1 = torch.from_numpy(gamma1_plan[idx_0, idx_1].reshape(-1, 1)).float().to(device)

        delta_t = t_train[t + 1] - t_train[t]
        t_relative = torch.rand(x0.shape[0], 1).type_as(x0)
        t_samp = delta_t * t_relative

        xt_samp, gt_samp, ut_samp, masst_samp, index = compute_xt_ut_gt(
            t_relative, delta_t, x0, x1, mass0, mass1, delta
        )

        ts.append(t_samp[index] + t_train[t])
        xts.append(xt_samp)
        uts.append(ut_samp)
        gts.append(gt_samp)
        massts.append(masst_samp)

    return torch.cat(ts), torch.cat(xts), torch.cat(uts), torch.cat(gts), torch.cat(massts)


def compute_trajectories_action(df, f_net, device, all_times, sigma, use_mass, delta, num_points=None, num_runs=1):
    del sigma
    start_time = float(np.min(all_times))
    end_time = float(np.max(all_times))
    data = torch.tensor(df[df["samples"] == start_time].values, dtype=torch.float32).requires_grad_()
    x0 = data[:, 1:].to(device).requires_grad_()

    results = []
    for run_idx in tqdm(range(num_runs)):
        seed = run_idx
        random.seed(seed)
        np.random.seed(seed)
        torch.random.manual_seed(seed)
        if num_points is not None:
            sample_indices = random.sample(range(x0.size(0)), num_points)
            x0_subset = x0[sample_indices].to(device)
        else:
            x0_subset = x0.to(device)

        lnw0 = torch.log(torch.ones(x0_subset.shape[0], 1)).to(device)
        initial_state = (x0_subset, lnw0)

        for param in f_net.parameters():
            param.requires_grad = False

        ts = torch.linspace(start_time, end_time, 100, device=device)
        traj, traj_lnw = odeint(ODEFunc2(f_net), initial_state, ts)

        if use_mass:
            weight = torch.exp(traj_lnw)
        else:
            weight = torch.ones_like(traj_lnw)

        dt = ts[1:] - ts[:-1]
        action = 0.0
        for time_idx in range(len(ts) - 1):
            this_t = ts[time_idx]
            this_x = traj[time_idx]
            this_m = weight[time_idx]
            this_vel = f_net.v_net(this_t, this_x)
            loss_vel = 0.5 * torch.sum((this_vel**2).sum(dim=1, keepdim=True) * this_m)

            this_g = f_net.g_net(this_t, this_x)
            loss_g = 0.5 * (delta**2) * torch.sum((this_g**2) * this_m)

            action += (loss_vel + loss_g) * dt[time_idx]

        action_value = action.detach().cpu().item() / x0_subset.shape[0]
        results.append(
            {
                "Model": f"WFR-FM_run{run_idx + 1}",
                "action": action_value,
            }
        )

    return results
