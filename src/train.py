__all__ = ["pretrain", "train"]

import logging

import numpy as np
import torch
from tqdm import tqdm

from utils import compute_uot_plans, get_batch


def pretrain(
    model,
    df,
    optimizer_1,
    optimizer_2,
    scheduler_1=None,
    scheduler_2=None,
    n_epoch=1000,
    test_interval=100,
    batch_size=256,
    hold_one_out=False,
    hold_out="random",
    logger=None,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    save_dir=None,
    relative_mass=None,
    delta=1.0,
    use_mini_batch=False,
    chunk_size=2000,
    ema=None,
):
    time_labels = df["samples"].to_numpy()
    data = df.iloc[:, 1:].to_numpy()
    x_all = [data[time_labels == t] for t in np.unique(time_labels)]
    x_selected = [data[time_labels == v] for v in np.unique(time_labels) if v != hold_out]
    t_train = [v for v in np.unique(time_labels) if v != hold_out]

    if logger is not None:
        logger.info("Begin flow and growth matching")

    gamma0_plans, gamma1_plans, sampling_info = compute_uot_plans(
        x_selected,
        t_train,
        delta=delta,
        draw=False,
        use_mini_batch_uot=use_mini_batch,
        chunk_size=chunk_size,
    )

    progress_bar = tqdm(range(n_epoch), desc="Begin flow and growth matching...", unit="epoch")
    vloss_list = []
    gloss_list = []
    loss_list = []

    for epoch in progress_bar:
        optimizer_1.zero_grad()
        optimizer_2.zero_grad()

        t, xt, ut, gt, masst = get_batch(
            x_selected,
            t_train,
            batch_size,
            gamma0_plans,
            gamma1_plans,
            delta,
            relative_mass,
            sampling_info,
        )
        vt = model.v_net(t, xt)
        gt_pred = model.g_net(t, xt)

        vloss = torch.mean((vt - ut) ** 2 * masst)
        gloss = torch.mean((gt_pred - gt) ** 2 * masst)
        loss = vloss + gloss

        vloss_list.append(vloss.item())
        gloss_list.append(gloss.item())
        loss_list.append(loss.item())

        loss.backward()

        optimizer_1.step()
        optimizer_2.step()
        if scheduler_1 is not None:
            scheduler_1.step()
        if scheduler_2 is not None:
            scheduler_2.step()
        if ema is not None:
            ema.update()

        if logger is not None:
            logger.info(
                "Epoch %d: loss=%.6f, vloss=%.6f, gloss=%.6f",
                epoch,
                loss.item(),
                vloss.item(),
                gloss.item(),
            )
        else:
            logging.info(
                "Epoch %d: loss=%.6f, vloss=%.6f, gloss=%.6f",
                epoch,
                loss.item(),
                vloss.item(),
                gloss.item(),
            )
        progress_bar.set_postfix(
            {
                "loss": f"{loss.item():.6f}",
                "vloss": f"{vloss.item():.6f}",
                "gloss": f"{gloss.item():.6f}",
            }
        )

    return model, vloss_list, gloss_list, loss_list


def train(
    model,
    df,
    n_epoch=1000,
    test_interval=100,
    batch_size=256,
    hold_one_out=False,
    hold_out="random",
    logger=None,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    save_dir=None,
    relative_mass=None,
    delta=1.0,
    use_mini_batch=False,
    chunk_size=2000,
    lr_v=0.005,
    lr_g=0.005,
    eta_min=1e-5,
    ema=None,
):
    optimizer_1 = torch.optim.Adam(model.v_net.parameters(), lr=lr_v)
    optimizer_2 = torch.optim.Adam(model.g_net.parameters(), lr=lr_g)
    scheduler_1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_1, T_max=n_epoch, eta_min=eta_min)
    scheduler_2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_2, T_max=n_epoch, eta_min=eta_min)

    return pretrain(
        model,
        df,
        optimizer_1,
        optimizer_2,
        scheduler_1=scheduler_1,
        scheduler_2=scheduler_2,
        n_epoch=n_epoch,
        test_interval=test_interval,
        batch_size=batch_size,
        hold_one_out=hold_one_out,
        hold_out=hold_out,
        logger=logger,
        device=device,
        save_dir=save_dir,
        relative_mass=relative_mass,
        delta=delta,
        use_mini_batch=use_mini_batch,
        chunk_size=chunk_size,
        ema=ema,
    )
