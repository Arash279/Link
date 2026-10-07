# HP_30 Workflow

This is a parallel workflow for the HP_30 database.

- Same extraction logic as the main workflow
- Same DE + LS fitting algorithm
- Same fixed priors and formula-derived initial-parameter logic
- Only the default database target changes from `AP_1p5.db` to `AP_30.db`
- Console logging is intentionally more detailed for testing and inspection

## Default database

```text
D:\Desktop\EE5003\data\AP_30.db
```

## Main entry

```powershell
python D:\Desktop\LinkCodex\baseline1\hp30_workflow\run_workflow.py --stage all --table exp_10 --seed 0
```

## Performance validation

```powershell
python D:\Desktop\LinkCodex\baseline1\hp30_workflow\validate_performance.py --table exp_10 --seed 0
```
