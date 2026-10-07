# Baseline1 Workflow

This folder is the current clean workflow for the motor impedance project.
The older `baselines/` folder is treated as deprecated history; `baseline1/CurVer.py`
remains the algorithm reference for the current DE+LS method.

## Structure

1. `extraction.py`
   - Reads impedance tables from SQLite.
   - Runs the full `Parameter_Fitting` stage.
   - Extracts `fr/fa/Zmax/Zanti`, leakage `L_sigma`, `Csf^HF/Csf^LF`, `Rrs`, and `Lm`.
   - Writes a consolidated `stage1_initial_inputs.json/csv` for the fit stage.

2. `model.py`
   - Re-exports the current equivalent-circuit model from `baseline1/CurVer.py`.
   - Keeps the model equations and parameter vector unchanged.

3. `fitting.py`
   - Calls the existing DE+LS implementation from `CurVer.py`.
   - Saves optimized parameters, metrics, and candidate summaries.
   - Builds the fit initial point from stage-1 parameter-fitting outputs plus a few fixed priors.

4. `validation.py`
   - Loads fitted parameters.
   - Evaluates raw-space metrics on the full table.
   - Optionally runs GP residual analysis.

5. `run_workflow.py`
   - Single CLI entry for the three stages.

## Commands

Run all stages:

```powershell
python -m baseline1.workflow.run_workflow --stage all --table exp_10 --seed 0
```

Run only extraction:

```powershell
python -m baseline1.workflow.run_workflow --stage extract
```

Run only fitting:

```powershell
python -m baseline1.workflow.run_workflow --stage fit --table exp_10 --seed 0
```

Run validation with an existing parameter file:

```powershell
python -m baseline1.workflow.run_workflow --stage validate --table exp_10 --params-path baseline1\workflow_outputs\stage2_fit_params_exp_10_seed0.json
```

Default outputs are written to:

```text
baseline1/workflow_outputs/
```

## Notes

- The fitting algorithm is not changed. The workflow only separates configuration,
  data loading, model access, fitting orchestration, and validation.
- Stage 1 now derives:
  `Lls/Llr`, `Lm`, `Rrs`, `CsfHF`, `CsfLF`, `fr`, `Zmax`, `fa`, and `Zanti`
  from impedance tables through the `Parameter_Fitting` logic.
- The remaining fixed priors are:
  `hp`, `connection`, `nLls`, and `Lad`.
- `Csw`, `Rsw`, `Rsf`, `Csf0`, and `Rcore` are still computed analytically in
  `initial_params.py`.
- The paper formula for `eta_Lls` is retained in code for reference, but it is not
  injected into `nLls` because that mapping does not match the present CurVer model.
- Keep `CurVer.py` as the reference implementation until the model is fully promoted
  into standalone modules.
- Avoid editing deprecated `baselines/` unless reproducing old comparisons.
