# Mag/Phase Workflow

This workflow is the true contrast group for the current Re/Im fitting method.

It keeps the same data loading, extraction reports, equivalent-circuit model,
DE+LS optimizer structure, parameter bounds, sampling, and validation style.
Only the stage-2 fitting residual changes:

```text
residual = [log10(|Z_sim|) - log10(|Z_exp|), phase(Z_sim) - phase_exp]
```

The phase residual is wrapped to `(-180, 180]` and scaled with the same default
idea used in older scripts: `30 deg ~= 1 residual unit`.

## Run

```powershell
C:\Users\35789\miniconda3\envs\common\python.exe -m baseline1.mag_phase_workflow.run_experiment --stage all --table exp_10 --seed 0 --no-gp
```

Validate the last saved parameters:

```powershell
C:\Users\35789\miniconda3\envs\common\python.exe -m baseline1.mag_phase_workflow.validate_performance --table exp_10 --seed 0
```

Outputs are written to:

```text
baseline1/mag_phase_workflow/outputs/
```

