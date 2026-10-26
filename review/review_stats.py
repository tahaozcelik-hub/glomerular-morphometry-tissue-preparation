"""Review categories by preparation, for flagged and unflagged glomeruli (Fisher's exact tests)."""
import os
import sys

import pandas as pd
from scipy.stats import fisher_exact

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RES  # noqa: E402

OUTCOMES = {"ischaemic/sclerotic": ["ischaemic", "global", "segmental"],
            "uncertain or artefact": ["uncertain", "artefact"],
            "global sclerosis": ["global"]}


def main():
    k = pd.read_csv(os.path.join(RES, "histology_review.csv"))
    rows = []
    for st in ("flag", "control"):
        for name, cats in OUTCOMES.items():
            a = k[(k.set == st) & (k.prep == "FFPE")].category.isin(cats)
            b = k[(k.set == st) & (k.prep == "fresh_frozen")].category.isin(cats)
            _, p = fisher_exact([[a.sum(), len(a) - a.sum()], [b.sum(), len(b) - b.sum()]])
            rows.append(dict(set=st, outcome=name, FFPE=f"{a.sum()}/{len(a)} ({100 * a.mean():.0f}%)",
                             frozen=f"{b.sum()}/{len(b)} ({100 * b.mean():.0f}%)", fisher_p=p))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "histology_review_summary.csv"), index=False)


if __name__ == "__main__":
    main()
