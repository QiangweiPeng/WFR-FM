from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str
    data_file: str
    dim: int
    delta: float
    batch_size: int
    n_epoch: int
    hidden_dim: int = 256
    n_hiddens: int = 5
    seed: int = 11
    lr_v: float = 0.005
    lr_g: float = 0.005
    eta_min: float = 1e-5
    hold_out: float | None = None
    use_mini_batch: bool = False
    chunk_size: int = 2000
    ema_decay: float | None = 0.995
    apply_ema_for_eval: bool = True
    plot_comparison: bool = True
    plot_growth_per_time: bool = False
    reducer_path: str | None = None
    plot_transparent: bool = True
    report_growth_correlation: bool = False


EXPERIMENTS = {
    "eb_100d": ExperimentConfig(
        experiment_name="eb_100d_scaled",
        data_file="data/eb_noscale.csv",
        dim=100,
        delta=35.0,
        batch_size=512,
        n_epoch=3000,
        use_mini_batch=True,
        chunk_size=2000,
        ema_decay=0.995,
        apply_ema_for_eval=True,
    ),
    "eb_50d": ExperimentConfig(
        experiment_name="eb_50d",
        data_file="data/eb_noscale.csv",
        dim=50,
        delta=25.0,
        batch_size=512,
        n_epoch=3000,
        use_mini_batch=True,
        ema_decay=0.995,
        apply_ema_for_eval=True,
    ),
    "eb_5d": ExperimentConfig(
        experiment_name="eb_5d",
        data_file="data/eb_5dim.csv",
        dim=5,
        delta=2.0,
        batch_size=512,
        n_epoch=3000,
        use_mini_batch=True,
        ema_decay=0.995,
        apply_ema_for_eval=True,
        plot_transparent=False,
    ),
    "simulation": ExperimentConfig(
        experiment_name="simulation_new",
        data_file="data/simulation_gene_data.csv",
        dim=2,
        delta=1.5,
        batch_size=400,
        n_epoch=3000,
        use_mini_batch=False,
        ema_decay=0.995,
        apply_ema_for_eval=True,
        report_growth_correlation=True,
    ),
    "cite_50d": ExperimentConfig(
        experiment_name="cite_50d_new",
        data_file="data/cite_pca50.csv",
        dim=50,
        delta=30.0,
        batch_size=256,
        n_epoch=3000,
        use_mini_batch=True,
        ema_decay=0.995,
        apply_ema_for_eval=True,
    ),
    "dygen_5d": ExperimentConfig(
        experiment_name="dygen_5d",
        data_file="data/dygen.csv",
        dim=5,
        delta=2.0,
        batch_size=256,
        n_epoch=3000,
        seed=42,
        use_mini_batch=False,
        ema_decay=0.995,
        apply_ema_for_eval=True,
    ),
    "gaussian_1000d": ExperimentConfig(
        experiment_name="gaussian_1000d_new",
        data_file="data/gaussian_1000d.csv",
        dim=1000,
        delta=1.4,
        batch_size=256,
        n_epoch=3000,
        lr_v=1e-3,
        lr_g=1e-3,
        use_mini_batch=False,
        ema_decay=None,
        apply_ema_for_eval=False,
    ),
    "emt": ExperimentConfig(
        experiment_name="emt",
        data_file="data/emt.csv",
        dim=10,
        delta=2.0,
        batch_size=256,
        n_epoch=3000,
        use_mini_batch=False,
        ema_decay=0.995,
        apply_ema_for_eval=True,
    ),
    "weinreb_50d": ExperimentConfig(
        experiment_name="weinreb_50d",
        data_file="data/Weinreb_alltime.csv",
        dim=50,
        delta=15.0,
        batch_size=512,
        n_epoch=3000,
        use_mini_batch=True,
        ema_decay=0.995,
        apply_ema_for_eval=True,
        plot_growth_per_time=True,
        reducer_path="results/weinreb_50d/umap_weinreb.pkl",
    ),
}
