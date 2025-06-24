"""
--------------------------------------------------------------------------------
<reader project>
src/reader/utils/prune_config.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from reader.config import ReaderCfg

def _prune_empty_config_elements(df, cfg: ReaderCfg):
    valid = set(df["genotype"].dropna().unique())

    # 1) trim your cosmetic-renames map
    if cfg.naming.use_short:
        cfg.naming.map = {
            k: v for k, v in cfg.naming.map.items()
            if v in valid
        }

    # helper to prune a list of (label, [values]) or {label: [values]}
    def prune_groups(gr_list):
        pruned = []
        for grp in gr_list:
            # dict style: {label: vals}
            if isinstance(grp, dict) and len(grp) == 1:
                lbl, vals = next(iter(grp.items()))
                if set(vals) & valid:
                    pruned.append(grp)

            # tuple/list style: (label, vals)
            elif isinstance(grp, (list, tuple)) and len(grp) == 2:
                lbl, vals = grp
                if set(vals) & valid:
                    pruned.append(grp)

            # otherwise skip
        return pruned

    # 2) trim defaults
    dft = cfg.plotting.defaults
    dft.groups = prune_groups(dft.groups)

    # 3) trim any per-plot override
    for spec in cfg.plotting.plots:
        if getattr(spec, "groups", None):
            spec.groups = prune_groups(spec.groups)
