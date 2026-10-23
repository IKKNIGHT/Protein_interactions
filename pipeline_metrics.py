"""Shared hit / self-consistency classification for the PPI-binder-design
pipeline.

CANONICAL SOURCE: Phase9_Cross_Target_Analysis.ipynb cell 7, where this
formula first appears:

    d['hit'] = (d.pose_rmsd <= 8) & (d.com_dist <= 10)

Phase9 cell 8 (the MASTER data export) independently redefined the exact
same two thresholds as `self_consistent`/`hit` -- a genuine duplicate with
no numeric difference. Both call sites now import from here instead.

Why Phase6_8_Sweep_Eval.ipynb cell 13's `trusted` gate was NOT folded into
this module: `trusted = (f1 >= 0.70) & (com_dist <= 10) & (target_fit <= 8)`
measures something different in kind, not just in threshold. `hit`/
`self_consistent` (here) are per-DESIGN outcome metrics -- did this one
designed binder's predicted pose land close to where RFdiffusion put it,
and close to the native partner's site. `trusted` is a per-TARGET inclusion
gate -- was this target's native-complex control itself reproduced well
enough by the structure predictor to trust any of its downstream design
results at all. It only coincidentally reuses the value 10 for its own
`com_dist` check; forcing it through `classify_design()` would misrepresent
what it measures (it isn't classifying a design as a hit, it's deciding
whether to trust a target's entire evaluation). Only the shared numeric
constant (`COM_DIST_MAX`) is reused there, not the function -- see that
cell's comment for the exact reasoning.

Phase6_8 cell 14's `pass` filter (iptm/PAE-based) is unrelated to all of
this and is left untouched.
"""

POSE_RMSD_MAX = 8.0   # A: RFdiffusion backbone vs. Boltz-repredicted pose
COM_DIST_MAX = 10.0   # A: predicted binder center-of-mass vs. native partner site


def classify_design(pose_rmsd, com_dist):
    """Classify one designed binder's outcome.

    Returns (hit, self_consistent), matching Phase9 cell 7/8's original
    inline formula exactly (zero numeric change):
      self_consistent = pose_rmsd <= POSE_RMSD_MAX
      hit             = self_consistent & (com_dist <= COM_DIST_MAX)

    Works on scalars or on pandas Series/numpy arrays alike (plain boolean
    operators, no pandas-specific calls), so existing call sites that map
    this over a DataFrame column don't need to change their style.
    """
    self_consistent = pose_rmsd <= POSE_RMSD_MAX
    hit = self_consistent & (com_dist <= COM_DIST_MAX)
    return hit, self_consistent
