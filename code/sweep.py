"""
Sensitivity sweep for the debrief rules (documentation, sections 4 and 6). Not needed to reproduce
submission.csv -- helper only.

    python code/sweep.py

Prints, for each setting, the composition of the 500 and its overlap with the default shortlist.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main as M  # noqa: E402

prep = M.prepare(".")
base_ids, base_stats = M.shortlist(prep)
base = set(base_ids)
print("\nDEFAULT:", M.DEFAULTS)
print("  composition", base_stats["composition"], "| bonus pts", base_stats["bonus_pts"])

GRID = [
    ("pcc_full", [12, 18, 24, 36]),
    ("pcc_low", [5, 10, 15]),
    ("pcc_q", [0.50, 0.65, 0.75, 0.85]),
    ("star_tech", [80, 85, 90]),
    ("star_q", [0.50, 0.70, 0.85]),
    ("star_role_fit", [True, False]),
    ("old_boys_bonus", [0.0, 2.5, None]),
]
print("\n%-16s %-8s %-9s %-14s %-12s %-8s %-8s" % ("param", "value", "overlap", "fast_tracked", "partial", "stars", "old_boys"))
for name, values in GRID:
    for v in values:
        ids, st = M.shortlist(prep, **{name: v})
        c = st["composition"]
        print("%-16s %-8s %4d/500  %-14d %-12d %-8d %-8d" % (
            name, v, len(base & set(ids)), c["fast_tracked"], c["partial_ramp"], c["new_college_stars"], c["old_boys"]))
