# Experimental Re/Im Workflow

This folder contains an experimental workflow variant. It keeps the same
three-stage structure as `baseline1/workflow`:

1. parameter extraction,
2. equivalent-circuit fitting,
3. validation.

Only stage 2 is owned by this experimental folder. The measured data are first
converted into complete complex impedance:

```text
Z = |Z| * exp(j * phase)
```

The optimizer then fits:

```text
Re(Z_sim - Z_data), Im(Z_sim - Z_data)
```

Important: the current `baseline1/CurVer.py` method already uses Re/Im residuals
internally. This folder is therefore an isolated workflow variant for testing and
bookkeeping, not yet a numerical contrast against a magnitude+phase baseline.

The equivalent-circuit model, DE+LS strategy, bounds, sampling, weighting, and
validation logic remain aligned with the current baseline configuration.

## Run

```powershell
C:\Users\35789\miniconda3\envs\common\python.exe -m baseline1.experimental_reim_workflow.run_experiment --stage all --table exp_10 --seed 0 --no-gp
```

Outputs are written to:

```text
baseline1/experimental_reim_workflow/outputs/
```
