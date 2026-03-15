from __future__ import annotations

from collections.abc import Mapping

from reader.core.registry import Plugin, PluginConfig
from reader.core.workbench import PluginSemantics
from reader.plugins.transform._labeling import apply_label_mappings


class AssayLabelsCfg(PluginConfig):
    refs: list[str] | None = None
    in_place: bool = False
    case_insensitive: bool = True
    suffix: str = "_alias"


class AssayLabelsTransform(Plugin):
    key = "assay_labels"
    category = "transform"
    semantics = PluginSemantics(
        category="transform",
        domain="assay",
        family="label_enrichment",
        summary="Materialize configured assay.labels into dataframe columns.",
        tags=("assay", "labels", "annotation"),
    )
    ConfigModel = AssayLabelsCfg

    @classmethod
    def input_contracts(cls) -> Mapping[str, str]:
        return {"df": "tidy.v1"}

    @classmethod
    def output_contracts(cls) -> Mapping[str, str]:
        return {"df": "tidy.v1"}

    @classmethod
    def output_contract_surfaces(cls) -> Mapping[str, object]:
        return cls.passthrough_output_contract_surfaces(
            passthrough={"df": "df"},
            promoted_examples={"df": ("plate_reader.annotated.v1",)},
        )

    def resolve_output_contracts(self, *, inputs, outputs, cfg, where):
        del cfg
        return self.inherit_dataframe_output_contracts(
            inputs=inputs,
            outputs=outputs,
            passthrough={"df": "df"},
            where=where,
        )

    def run(self, ctx, inputs, cfg: AssayLabelsCfg):
        assay = ctx.assay or {}
        label_specs = (assay.get("labels") or {}) if isinstance(assay, Mapping) else {}
        refs = cfg.refs or list(label_specs)
        if not refs:
            raise ValueError("assay_labels: no assay.labels are configured")

        mappings: dict[str, Mapping[str, str]] = {}
        output_names: dict[str, str] = {}
        for ref in refs:
            label_spec = label_specs.get(ref)
            if label_spec is None:
                raise ValueError(f"assay_labels: assay.labels missing key '{ref}'")
            if hasattr(label_spec, "model_dump"):
                label_spec = label_spec.model_dump()
            if not isinstance(label_spec, Mapping):
                raise ValueError(f"assay_labels: assay.labels.{ref} must resolve to a mapping")
            source = str(label_spec.get("source") or "").strip()
            if not source:
                raise ValueError(f"assay_labels: assay.labels.{ref}.source must be a non-empty string")
            values = label_spec.get("values", {}) or {}
            if not isinstance(values, Mapping):
                raise ValueError(f"assay_labels: assay.labels.{ref}.values must be a mapping")
            mappings[source] = {str(k): str(v) for k, v in values.items()}
            output = label_spec.get("output")
            output_names[source] = (
                str(output) if isinstance(output, str) and output.strip() else f"{source}{cfg.suffix}"
            )

        return {
            "df": apply_label_mappings(
                ctx=ctx,
                df=inputs["df"],
                mappings=mappings,
                output_names=output_names,
                in_place=cfg.in_place,
                case_insensitive=cfg.case_insensitive,
                label="assay_labels",
            )
        }
