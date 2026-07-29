"""
--------------------------------------------------------------------------------
<reader project>
src/reader/plugins/transform/ratio.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from contextlib import suppress

import numpy as np
import pandas as pd
from pydantic import Field

from reader.workbench.ports import dataframe_input, dataframe_output
from reader.workbench.registry import Plugin, PluginConfig


class RatioCfg(PluginConfig):
    name: str
    numerator: str
    denominator: str
    align_on: list[str] = Field(default_factory=lambda: ["position", "time"])


class RatioTransform(Plugin):
    ConfigModel = RatioCfg

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

    def run(self, ctx, inputs, cfg: RatioCfg):
        input_columns = list(inputs["df"].columns)
        df, emit_value_provenance = _with_value_provenance(inputs["df"])

        # Build alignment key; auto-augment with per-sheet/scope cols if present
        key = [c for c in cfg.align_on if c in df.columns]
        for extra in ("sheet_index", "sheet_name", "source"):
            if extra in df.columns and extra not in key:
                key.append(extra)
        if not key:
            available = sorted(set(df.columns))
            raise ValueError(
                "ratio: none of align_on columns are present in the input.\n"
                f"  align_on: {cfg.align_on}\n"
                f"  available: {available}"
            )

        # Partition numerator/denominator; keep ALL metadata on numerator side
        lhs = (
            df[df["channel"] == cfg.numerator]
            .rename(
                columns={
                    "value": "__num__",
                    "value_policy_clipped": "__num_policy_clipped__",
                    "value_instrument_overflow": "__num_instrument_overflow__",
                    "value_bound_kind": "__num_bound_kind__",
                }
            )
            .copy()
        )
        rhs = (
            df[df["channel"] == cfg.denominator]
            .rename(
                columns={
                    "value": "__den__",
                    "value_policy_clipped": "__den_policy_clipped__",
                    "value_instrument_overflow": "__den_instrument_overflow__",
                    "value_bound_kind": "__den_bound_kind__",
                }
            )
            .copy()
        )
        if lhs.empty or rhs.empty:
            available = sorted(df["channel"].dropna().astype(str).unique().tolist())
            missing = []
            if lhs.empty:
                missing.append(cfg.numerator)
            if rhs.empty:
                missing.append(cfg.denominator)
            raise ValueError(
                f"ratio: requested channel(s) missing from input.\n  missing: {missing}\n  available: {available}"
            )

        # Keep only join keys + denominator on RHS to avoid suffix collisions
        rhs = rhs[
            key
            + [
                "__den__",
                "__den_policy_clipped__",
                "__den_instrument_overflow__",
                "__den_bound_kind__",
            ]
        ]

        # Join (lhs may be many-to-one vs rhs on the key)
        merged = pd.merge(lhs, rhs, on=key, how="inner", validate="many_to_one")

        # Numerics + validity filter (drop invalids to satisfy tidy.v1: no NaNs)
        merged["__num__"] = pd.to_numeric(merged["__num__"], errors="coerce")
        merged["__den__"] = pd.to_numeric(merged["__den__"], errors="coerce")
        ok = merged["__num__"].notna() & merged["__den__"].notna() & (merged["__den__"] != 0)
        dropped = int((~ok).sum())
        if dropped:
            ctx.logger.warning(
                "[warn]ratio[/warn] • %s: dropped %d row(s) due to missing/zero denominator", cfg.name, dropped
            )

        merged = merged.loc[ok].copy()
        bounded = merged["__num_bound_kind__"].ne("exact") | merged["__den_bound_kind__"].ne("exact")
        nonpositive = merged["__num__"].le(0.0) | merged["__den__"].le(0.0)
        if (bounded & nonpositive).any():
            raise ValueError("ratio: bounded values require positive operands for directional bound propagation")
        merged["value"] = merged["__num__"] / merged["__den__"]
        merged["channel"] = cfg.name
        merged["value_policy_clipped"] = merged["__num_policy_clipped__"] | merged["__den_policy_clipped__"]
        merged["value_instrument_overflow"] = (
            merged["__num_instrument_overflow__"] | merged["__den_instrument_overflow__"]
        )
        denominator_bounds = merged["__den_bound_kind__"].map(
            {"exact": "exact", "lower": "upper", "upper": "lower", "indeterminate": "indeterminate"}
        )
        merged["value_bound_kind"] = [
            _combine_bounds(numerator, denominator)
            for numerator, denominator in zip(merged["__num_bound_kind__"], denominator_bounds, strict=True)
        ]
        if emit_value_provenance and "overflow" in merged.columns:
            merged["overflow"] = merged["value_instrument_overflow"]

        # Restore original column set in original order (inherits metadata from lhs)
        derived = merged[df.columns].copy()

        out = pd.concat([df, derived], ignore_index=True)
        if not emit_value_provenance:
            # Generic ratios remain usable, but missing provenance is not evidence
            # that an observation is exact. Response-window ingestion rejects it.
            out = out.loc[:, input_columns]

        with suppress(Exception):
            ctx.logger.info(
                "ratio • [accent]%s[/accent] = %s / %s • +%d row(s) • keys=%s",
                cfg.name,
                cfg.numerator,
                cfg.denominator,
                len(derived),
                key,
            )

        return {"df": out}


def _with_value_provenance(frame: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    result = frame.copy()
    explicit_fields = {"value_policy_clipped", "value_instrument_overflow", "value_bound_kind"}
    present = explicit_fields & set(result.columns)
    if present and present != explicit_fields:
        raise ValueError("ratio: value provenance must provide all three explicit fields together")
    if present:
        policy_clipped = _strict_boolean(result["value_policy_clipped"], field="value_policy_clipped")
        instrument_overflow = _strict_boolean(result["value_instrument_overflow"], field="value_instrument_overflow")
        bounds = result["value_bound_kind"]
        if bounds.isna().any() or not bounds.map(lambda value: isinstance(value, str)).all():
            raise ValueError("ratio: value_bound_kind provenance must contain strings without missing values")
        bounds = bounds.astype(str)
        allowed = {"exact", "lower", "upper", "indeterminate"}
        unknown = sorted(set(bounds) - allowed)
        if unknown:
            raise ValueError(f"ratio: unsupported value_bound_kind values: {unknown}")
        affected = policy_clipped | instrument_overflow
        if not affected.eq(bounds.ne("exact")).all():
            raise ValueError("ratio: clipping and overflow provenance disagrees with value_bound_kind")
        if "overflow" in result.columns:
            observed_overflow = _strict_boolean(result["overflow"], field="overflow")
            if not observed_overflow.eq(instrument_overflow).all():
                raise ValueError("ratio: overflow disagrees with explicit instrument-overflow provenance")
    else:
        policy_clipped = pd.Series(False, index=result.index, dtype=bool)
        instrument_overflow = pd.Series(False, index=result.index, dtype=bool)
        bounds = pd.Series("exact", index=result.index, dtype=object)
    result["value_policy_clipped"] = policy_clipped
    result["value_instrument_overflow"] = instrument_overflow
    result["value_bound_kind"] = bounds
    return result, bool(present)


def _strict_boolean(values: pd.Series, *, field: str) -> pd.Series:
    if values.isna().any() or not values.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise ValueError(f"ratio: {field} provenance must contain booleans without missing values")
    return values.astype(bool)


def _combine_bounds(left: object, right: object) -> str:
    bounds = {str(left), str(right)} - {"exact"}
    if not bounds:
        return "exact"
    if len(bounds) == 1:
        return bounds.pop()
    return "indeterminate"
