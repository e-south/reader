from __future__ import annotations

from reader.core.labeling import apply_label_mappings
from reader.workbench.ports import dataframe_input, dataframe_output
from reader.workbench.registry import Plugin, PluginConfig


class AssayLabelsCfg(PluginConfig):
    refs: list[str] | None = None
    in_place: bool = False
    case_insensitive: bool = True
    suffix: str = "_alias"


class AssayLabelsTransform(Plugin):
    ConfigModel = AssayLabelsCfg

    @classmethod
    def input_ports(cls):
        return {"df": dataframe_input("df", "tidy.v1")}

    @classmethod
    def output_ports(cls):
        return cls.passthrough_output_ports(
            outputs={"df": dataframe_output("df", "tidy.v1")},
            passthrough={"df": "df"},
            promoted_examples={"df": ("plate_reader.annotated.v1",)},
        )

    def resolve_output_ports(self, *, inputs, outputs, cfg, where):
        del cfg
        return self.inherit_dataframe_output_ports(
            inputs=inputs,
            outputs=outputs,
            passthrough={"df": "df"},
            where=where,
        )

    def run(self, ctx, inputs, cfg: AssayLabelsCfg):
        if ctx.experiment is None:
            raise ValueError("assay_labels requires experiment semantics in the run context")
        label_specs = ctx.experiment.assay.resolve_label_specs(cfg.refs)
        if not label_specs:
            raise ValueError("assay_labels: no assay.labels are configured")

        mappings: dict[str, dict[str, str]] = {}
        output_names: dict[str, str] = {}
        for spec in label_specs:
            mappings[spec.source] = dict(spec.values)
            output_names[spec.source] = spec.output or f"{spec.source}{cfg.suffix}"

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
