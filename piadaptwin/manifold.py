from dataclasses import dataclass, field

THETA_NAMES: tuple[str, ...] = ("C", "R", "Ra", "Vref", "V1", "k10", "ke0")


@dataclass(frozen=True)
class ThetaVector:
    C: float = 1.2
    R: float = 0.95
    Ra: float = 0.06
    Vref: float = 5.0
    V1: float = 4.27
    k10: float = 0.119
    ke0: float = 0.456

    def as_tuple(self) -> tuple[float, ...]:
        return (self.C, self.R, self.Ra, self.Vref, self.V1, self.k10, self.ke0)


@dataclass
class FlowConfig:
    in_channels: int = 4
    samples_per_window: int = 240
    enc_layers: int = 3
    enc_filters: int = 64
    kernel: int = 7
    hidden: int = 64
    deriv_layers: int = 4
    state_dim: int = 4
    grid_points: int = 12
    window_seconds: float = 120.0
    horizons: tuple[int, ...] = (30, 60, 120)
    rtol: float = 1e-3
    atol: float = 1e-4


@dataclass
class PkpdConfig:
    e0: float = 97.4
    emax: float = 92.4
    hill_n: float = 1.43
    ec50: float = 3.08
    k12: float = 0.112
    k13: float = 0.042
    k21: float = 0.055
    k31: float = 0.0033
    pk_steps: int = 40


@dataclass
class MeterConfig:
    bis_samples: int = 120
    stft_window: int = 8
    stft_hop: int = 2
    n_bands: int = 5
    patch_seconds: int = 10
    embed_dim: int = 128
    n_heads: int = 4
    depth: int = 2
    horizons: tuple[int, ...] = (30, 60, 120)


@dataclass
class GovernorConfig:
    n_params: int = 7
    update_interval: float = 15.0
    eps_sigma: float = 1e-4
    noise_scale: float = 0.01
    n_samples: int = 4
    lr: float = 5e-3
    steps: int = 1


@dataclass
class LossConfig:
    alpha: float = 0.12
    beta: float = 0.03
    lambda_bis: float = 0.4
    mu: float = 0.08
    mu_missing: float = 0.01


@dataclass
class DataConfig:
    source: str = "vitaldb"
    manifest: str = ""
    fs: int = 500
    window_seconds: float = 120.0
    samples_per_window: int = 240
    batch_size: int = 128
    n_cases: int = 64
    split: tuple[float, float, float] = (0.7, 0.15, 0.15)


@dataclass
class StageConfig:
    stage1_epochs: int = 50
    stage2_epochs: int = 20
    stage3_epochs: int = 10
    stage1_lr: float = 3e-4
    stage3_lr: float = 5e-5
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    seeds: tuple[int, ...] = (42, 137, 271)


@dataclass
class ExperimentConfig:
    name: str = "main"
    flow: FlowConfig = field(default_factory=FlowConfig)
    pkpd: PkpdConfig = field(default_factory=PkpdConfig)
    meter: MeterConfig = field(default_factory=MeterConfig)
    governor: GovernorConfig = field(default_factory=GovernorConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)
    stages: StageConfig = field(default_factory=StageConfig)
    prior: ThetaVector = field(default_factory=ThetaVector)
    use_windkessel: bool = True
    use_pkpd: bool = True
    use_spectral: bool = True
    use_bayesadapt: bool = True
    use_neural_coupling: bool = True
    device: str = "auto"
    seed: int = 42
    out_dir: str = "runs/main"
