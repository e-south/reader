##  reader

A workbench for turning **raw instrument exports** into **tidy CSVs** and **plots**.

* **Plugin registry** for raw‑data **parsers** and **plotters**
* One **YAML config per experiment**
* Low‑friction **CLI** (`reader list / run`)
* Uses **uv** for deterministic dependency management

---

#### Workflow

```text
raw_data.xlsx ─┐
               │   (parser)        (QC + custom params)         (plotters)
plate_map.csv ─┴─► tidy_raw ─► merged ─► tidy_data.csv ─┬─► distributions.pdf
                                                        └─► time_series_*.pdf
```

Stages (see `reader/main.py`):

1. **load\_config** – YAML → `ReaderConfig`
2. **parse\_raw** – dispatch via registry (`reader.parsers.*`)
3. **parse\_plate\_map** – CSV/XLSX with `row`,`col` → tidy map
4. **merge** – inner‑join on `position`
5. **apply\_custom\_params** – blank subtraction, overflow handling, ratios
6. **plotting** – each spec in `plots:` is sent to `reader.plotters.*`

---

#### Project layout

```
reader/                    # repo root
├── src/reader/            # library code
│   ├── main.py            # CLI & orchestration
│   ├── config.py          # Pydantic schema helpers
│   ├── parsers/           # raw‑data parsers (plugin registry)
│   ├── processors/        # merge, QC, custom parameters
│   ├── plotters/          # plotting helpers (plugin registry)
│   └── utils/             # tiny helpers (fs, logging)
├── experiments/           # one folder per run (git‑ignored)
│   └── template/          # starter scaffold
├── tests/                 # pytest suites
├── uv.toml                # dependency lockfile
└── pyproject.toml
```

---

#### Installation

```bash
$ python -m venv .venv      # or use your preferred tool
$ source .venv/bin/activate
(.venv) $ uv sync           # installs locked deps + dev tools
(.venv) $ pip install -e .  # editable install of reader itself
```

#### Quick start

```bash
# 1. scaffold a new experiment (copies the template folder)
$ cp -r experiments/template_experiments/001_my_assay

# 2. take a look
$ tree -L 2 experiments/001_my_assay
.
├── config.yaml
├── raw_data/       # your .xlsx / .csv export
├── plate_map.csv   # well → metadata
└── outputs/        # created on first run

# 3. run
$ reader run 001     # or give the full path to config.yaml
✨ Done – tidy_data.csv and plots are in experiments/001_my_assay/outputs/
```

Need to see what’s available?

```bash
$ reader list
001_2025‑05‑12_biosensor_panel_M9_glu …
```




#### Minimal config example (`config.yaml`)

```yaml
author: "Eric South"
raw_data:    "./raw_data/PlateReader_export.xlsx"
plate_map:   "./plate_map.csv"
output_dir:  "./outputs/"

# which parser to use (see reader/parsers/)
data_parser: "synergy_h1_snapshot_and_timeseries"

# channels (columns) expected in the export
parameters: [OD600, CFP, YFP]

time_column: "Elapsed (s)"      # name inside the export
blank_correction: "avg_blank"   # or 'median_blank' / 'disregard'
overflow_action:  "max"         # 'max', 'min', 'zero', 'drop'

plots:
  - name: distributions
    module: distributions.py
    params:
      channels: [OD600, CFP, YFP]
    fig:
      seaborn_style: "ticks"
      palette: "colorblind"
```

Anything not specified falls back to defaults (see `ReaderConfig`).

---

#### Extending the pipeline

##### New raw parser

```text
src/reader/parsers/my_format.py
```

```python
from reader.parsers.raw import BaseRawParser, register_raw_parser

@register_raw_parser("my_format")
class MyParser(BaseRawParser):
    def parse(self) -> pd.DataFrame:
        # read self.path → tidy DataFrame with columns
        #   position, time, value, channel  (+ anything else you like)
        return df
```

Use it in YAML with `data_parser: "my_format"`.

##### New plot module

1. Drop a function `plot_whatever(df, blanks, output_dir, **kwargs)` in `src/reader/plotters/whatever.py`.
2. Add a plot spec in YAML:

```yaml
plots:
  - name: my_plot
    module: whatever.py
    params: { x: time, y: OD600, hue: treatment }
```