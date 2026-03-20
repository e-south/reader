# End-to-end demo

This walkthrough shows a typical flow from discovery to dataframe records, then plots/exports or notebooks. Prefer an explicit config path or experiment directory; `CONFIG|DIR|INDEX` works, but paths are deterministic and easier for agents to replay.

---

1) Find experiments

```bash
uv run reader ls
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
uv run reader explain ./experiments/2025/20250614_sensor_panel_M9_glu/config.yaml
```

3) Validate the config + inputs

```bash
uv run reader validate ./experiments/2025/20250614_sensor_panel_M9_glu/config.yaml
```

4) Run the pipeline (records only)

```bash
uv run reader run ./experiments/2025/20250614_sensor_panel_M9_glu/config.yaml
```

5) Inspect records

```bash
uv run reader records ./experiments/2025/20250614_sensor_panel_M9_glu/config.yaml
```

6) Generate plots

```bash
uv run reader plot ./experiments/2025/20250614_sensor_panel_M9_glu/config.yaml --list
uv run reader plot ./experiments/2025/20250614_sensor_panel_M9_glu/config.yaml
```

7) Generate exports

```bash
uv run reader export ./experiments/2025/20250614_sensor_panel_M9_glu/config.yaml --list
uv run reader export ./experiments/2025/20250614_sensor_panel_M9_glu/config.yaml
```

8) Scaffold a notebook

```bash
uv run reader notebook ./experiments/2025/20250614_sensor_panel_M9_glu/config.yaml
```

If you want a specific template, pass `--template <name>` (otherwise reader uses the first configured `notebooks.specs` entry or auto-picks).

See the [Notebooks guide](./notebooks.md) for opening and dependency setup.
