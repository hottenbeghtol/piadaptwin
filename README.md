# PI-AdapTwin

PI-AdapTwin is a research implementation of a physics-informed adaptive model for
hemodynamic and anesthetic-depth forecasting. It combines a neural ODE with a
two-element Windkessel constraint, a spectral network with pharmacokinetic and
pharmacodynamic constraints, and a diagonal-Gaussian parameter posterior.

## Current status

Clinical recordings, trained weights and dataset-specific loaders are not included.
The source configuration identifies a dataset but does not download or prepare it.
Integrating a loader requires preparing authorized recordings and verified targets in
`ClinicalBatch` format, including any required physiological parameters. The data
commands stop until a loader is implemented; forecasting, evaluation and export also
require a locally supplied trained checkpoint. Treatment outcome evaluation is not
implemented. This research code has not been clinically validated and must not be
used for clinical decisions.

The default pharmacokinetic and pharmacodynamic constants are representative values,
not individualized estimates. See [implementation limitations](docs/deviations.md).

## Installation

```sh
pip install -e .
```

Optional development tools can be installed with `pip install -e '.[dev]'`.

## Code layout

- `piadaptwin/intake`: input batch structure, channel definitions and data source interface.
- `piadaptwin/accumulator` and `piadaptwin/flowpath`: hemodynamic model and solver.
- `piadaptwin/metering`: spectral and pharmacokinetic components.
- `piadaptwin/governor`: posterior estimation and adaptation.
- `piadaptwin/commissioning`: training and checkpoint management.
- `piadaptwin/gauges`: evaluation metrics.
- `configs/experiment`: experiment configurations.

Run the available numerical and configuration checks with `python -m pytest`.
