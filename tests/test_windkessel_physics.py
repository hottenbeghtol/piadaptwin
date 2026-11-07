import torch

from piadaptwin.accumulator.windkessel import windkessel_residual


def test_residual_vanishes_on_exact_solution() -> None:
    compliance = 1.3
    resistance = 0.9
    dt = 1.0
    t = torch.arange(0, 16, dtype=torch.float64).reshape(1, -1)
    pressure = 70.0 + 0.5 * t
    cardiac = compliance * 0.5 + pressure / resistance
    residual = windkessel_residual(pressure, cardiac, compliance, resistance, dt)
    assert float(residual) < 1e-6


def test_residual_positive_on_mismatch() -> None:
    pressure = torch.randn(4, 12).double() * 5.0 + 80.0
    cardiac = torch.zeros(4, 12).double()
    residual = windkessel_residual(pressure, cardiac, 1.2, 0.95, 1.0)
    assert float(residual) > 0.0


def test_residual_differentiable_in_parameters() -> None:
    pressure = torch.randn(3, 12).double() * 4.0 + 85.0
    cardiac = torch.randn(3, 12).double().abs()
    compliance = torch.tensor(1.2, dtype=torch.float64, requires_grad=True)
    resistance = torch.tensor(0.9, dtype=torch.float64, requires_grad=True)
    residual = windkessel_residual(pressure, cardiac, compliance, resistance, 1.0)
    residual.backward()
    assert compliance.grad is not None and float(compliance.grad.abs()) > 0.0
    assert resistance.grad is not None and float(resistance.grad.abs()) > 0.0
