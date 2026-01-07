## reader demo

This walkthrough shows a typical flow from discovery to artifacts, then plots/exports or notebooks. Replace the index (`1`) with a config path or experiment directory as needed.

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

3) Validate the config + inputs

```bash
reader validate 1
```

4) Run the pipeline (artifacts only)

```bash
reader run 1
```

5) Inspect artifacts

```bash
reader artifacts 1
```

6) Generate plots

```bash
reader plot 1 --list
reader plot 1
```

7) Generate exports

```bash
reader export 1 --list
reader export 1
```

8) Scaffold a notebook

```bash
reader notebook 1
```

If you want a specific preset, pass `--preset <name>` (otherwise reader uses `notebook.preset` from config or auto-picks).

See [docs/notebooks.md](./notebooks.md) for opening and dependency setup.

---

@e-south
