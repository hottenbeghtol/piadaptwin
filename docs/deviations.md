# Implementation limitations

- Cardiac output for the Windkessel residual comes from a learned flow head.
- The implementation includes a two-element Windkessel model and a diagonal-Gaussian
  posterior; the three-element and full-covariance variants are not provided.
- Pharmacokinetic and pharmacodynamic constants use representative values and are not
  individualized estimates.
- Clinical recordings, dataset-specific loaders and trained weights are not included.
  The training and evaluation results cannot be reproduced without those inputs and
  appropriate data processing.
- ONNX export covers the waveform encoder only; the adaptive solver is not exported.
- The neural ODE uses a hand-written Dormand–Prince RK45 solver.
