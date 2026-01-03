## reader demo

This walkthrough shows a typical flow from discovery to deliverables and notebooks. Replace the index (`1`) with a config path or experiment directory as needed.


---

1) Find experiments

```bash
reader ls
```

Example output (shape only):

```text
Experiments
#  Name                         Outputs
1  20250512_sensor_panel_M9_glu  ✓
2  20250620_sensor_panel_crosstalk  —
```

2) Inspect the plan (no execution)

```bash
reader explain 1
```

3) Validate the config (no data I/O)

```bash
reader validate 1
```

4) Check file inputs

```bash
reader check-inputs 1
```
If any `reads: file:` paths are missing, the CLI points to the exact step and input key.

5) Run pipeline + deliverables

```bash
reader run 1
```

6) Deliverables only

```bash
reader deliverables 1 --list
reader deliverables 1
```

7) Scaffold a notebook

```bash
reader explore 1 --preset eda/basic
```

See [docs/notebooks.md](./notebooks.md) for opening and dependency setup.

8) Inspect artifacts

```bash
reader artifacts 1
```

---

@e-south
