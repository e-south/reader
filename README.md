[![CI](https://github.com/e-south/reader/actions/workflows/ci.yaml/badge.svg)](https://github.com/e-south/reader/actions/workflows/ci.yaml)

## reader

**reader** is a workbench for experimental data. The unit of work is an **experiment directory**: put raw inputs there, keep notebooks next to them, and write outputs in the same place. The CLI runs config-driven pipelines that produce **artifacts**, and optional **deliverables** (plots/exports) that render from those artifacts. Notebooks are for interactive exploration.

### Start here

- [Pipeline config + deliverables](./docs/pipeline.md)
- [CLI reference](./docs/cli.md)
- [Notebooks](./docs/notebooks.md)
- [Plugin development](./docs/plugins.md)
- [Spec / architecture](./docs/spec.md)
- [End-to-end demo](./docs/demo.md)

---

### Quickstart

Create an experiment directory:

```bash
mkdir -p experiments/my_experiment/{inputs,notebooks,outputs}
```

Run a config-driven pipeline (if `config.yaml` exists):

```bash
uv run reader explain experiments/my_experiment/config.yaml
uv run reader run     experiments/my_experiment/config.yaml   # pipeline + deliverables
```

Re-render deliverables or scaffold a notebook:

```bash
uv run reader deliverables experiments/my_experiment/config.yaml --list
uv run reader explore      experiments/my_experiment/config.yaml --preset eda/basic
```

---

### Install (short)

This repo uses **uv**. For full dev workflow, see [docs/spec.md](./docs/spec.md).

```bash
uv sync --locked
uv run reader --help
```

---

@e-south
