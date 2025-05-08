## reader

Plate reader analysis for the people!

reader/                              ← project root (uv workspace)
├── LICENSE
├── README.md                        ← project overview & quickstart
├── uv.toml                          ← uv workspace & project settings
├── pyproject.toml                   ← package build & dependencies
├── src/Reader/                      ← library code (parsers, processors, plotters, utils)
│   └── ...                          ← expand modules here
├── experiments/                     ← one subfolder per experiment
│   ├── exp_YYYY-MM-DD_<name>/
│   │   ├── config.yaml             ← raw_data, plate_map, output_dir, channels, etc.
│   │   ├── raw_data/               ← raw exports (CSV/.xlsx) **ignored by git**
│   │   ├── plate_map.csv           ← plate map CSV **ignored by git**
│   │   └── outputs/                ← generated plots, tidy_data.csv, logs **ignored**
│   └── ...                          ← add new experiments by copying template
└── tests/                           ← pytest test suites
    ├── conftest.py
    └── ...                          ← test_config.py, test_parsers, etc.

