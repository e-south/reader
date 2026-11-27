---
doc_id: reader-end-to-end-demo
surface: tutorial
owner: reader-maintainers
last_verified: 2026-11-27
summary: Create two small example workbooks, run a plate-reader pipeline, and verify the resulting records.
---

# Run a sample plate

Follow one well from an instrument workbook to a verified result. This example
uses invented values at three time points. You need [uv](https://docs.astral.sh/uv/)
and the [installed Reader CLI](./getting_started.md).

## Create the experiment

These shell commands start in a fresh temporary directory. Keep the same shell
open for the rest of the example.

```bash
work=$(mktemp -d)
cd "$work"
exp=./experiments/first_plate
reader init "$exp" --protocol plate_reader/single_reporter_screen
```

If you already created an empty `first_plate` from the README, you can use it
instead. Set `exp=./experiments/first_plate` in that workspace. The next block
writes the two example files and adds a display label to the config.

## Supply the inputs

The instrument workbook holds an OD600 reading and an RFP reading for well
`A1` at each time point. A separate workbook identifies the sample assigned to
that well. The label mapping turns its treatment code `reference` into the
display label `Reference`.

```bash
uv run --no-project --python 3.12 --with openpyxl --with pyyaml python - "$exp" <<'PY'
from pathlib import Path
import sys

import yaml
from openpyxl import Workbook

experiment = Path(sys.argv[1])
inputs = experiment / "inputs"

plate = Workbook()
sheet = plate.active
sheet.title = "Assay"
sheet.append(["Date", "2026-01-01"])
sheet.append(["Time", "12:00:00"])
for channel, values in [("OD600 B:600", (0.2, 0.4, 0.4)), ("RFP B:560,610", (10, 24, 24))]:
    sheet.append([channel])
    sheet.append([])
    sheet.append([None, "Time", "A1"])
    sheet.append([None, "00:00:00", values[0]])
    sheet.append([None, "00:10:00", values[1]])
    sheet.append([None, "00:20:00", values[2]])
plate.save(inputs / "kinetics.xlsx")

metadata = Workbook()
metadata.active.append(["position", "type", "design_id", "treatment"])
metadata.active.append(["A1", "SAMPLE", "example", "reference"])
metadata.save(inputs / "metadata.xlsx")

config_path = experiment / "config.yaml"
config = yaml.safe_load(config_path.read_text())
config["annotations"]["labels"] = {
    "treatment": {"source": "treatment", "output": "treatment_label", "values": {"reference": "Reference"}}
}
config_path.write_text(yaml.safe_dump(config, sort_keys=False))
PY
```

## Run and verify

```bash
reader validate "$exp"
reader run "$exp"
reader records "$exp"
reader verify "$exp"
```

Look for `ingest/df` in the record list. It contains the six input readings.
The final `sample_measurements/df` table also contains one RFP/OD600 ratio
per time point. Reader saves the tables under `outputs/artifacts/` and indexes
them in `outputs/manifests/records.json`. A successful `verify` confirms that
the current records match their inputs and saved files. Changing an input
after the run makes that discrepancy visible.

## Use the results

```bash
reader plot "$exp" --list
reader plot "$exp"
reader export "$exp" --list
reader notebook "$exp" --mode none
reader verify "$exp"
```

This configuration writes two PDFs under `outputs/plots/` and lists no exports.
The notebook command writes a Python scaffold under `outputs/notebooks/`.
See the [notebook guide](./notebooks.md) to open it or the [Python API](../core/python_api.md)
to read a verified table directly. For protocols with configured exports,
`reader export "$exp"` writes their files under the experiment's output directory.
