import torch

from piadaptwin.accumulator.windkessel import windkessel_residual
from piadaptwin.flowpath.integrator import dopri5_integrate


def test_windkessel_residual_golden_value() -> None:
    pressure = torch.tensor([[80.0, 82.0, 84.0, 86.0, 88.0]], dtype=torch.float64)
    cardiac = torch.zeros(1, 5, dtype=torch.float64)
    value = float(windkessel_residual(pressure, cardiac, 1.0, 1.0, 1.0))
    assert abs(value - 7398.6666666667) < 1e-4


def test_integrator_recovers_linear_solution() -> None:
    def field(t: float, z: torch.Tensor) -> torch.Tensor:
        return torch.full_like(z, 2.0)

    z0 = torch.zeros(1, 1, dtype=torch.float64)
    times = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    traj = dopri5_integrate(field, z0, times)
    assert abs(float(traj[0, 1, 0]) - 1.0) < 1e-6
    assert abs(float(traj[0, 2, 0]) - 2.0) < 1e-6
