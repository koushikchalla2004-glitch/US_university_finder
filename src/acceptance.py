from __future__ import annotations
from typing import Optional
from .utils import clamp, logit, inv_logit

# Personalized acceptance probability by anchoring on baseline admission rate
# and shifting the log-odds by profile match (GPA, GRE, IELTS, LOR/SOP). Heuristic.

def normalize_scores(cgpa: float, gre: Optional[float], ielts: Optional[float], lor_sop: float) -> dict:
    gpa = clamp((cgpa - 2.5)/(4.0-2.5), 0, 1)
    gre_n = None
    if gre is not None and gre > 0:
        # treat 290 as floor, 330+ as strong
        gre_n = clamp((gre - 290)/(340-290), 0, 1)
    ielts_n = None
    if ielts is not None and ielts > 0:
        ielts_n = clamp((ielts - 5.5)/(9.0-5.5), 0, 1)
    lor_n = clamp(lor_sop, 0, 1)
    return {"gpa": gpa, "gre": gre_n, "ielts": ielts_n, "lor": lor_n}

def acceptance_probability(baseline_rate: Optional[float], cgpa: float, gre: Optional[float], ielts: Optional[float], lor_sop: float) -> float:
    base = baseline_rate if baseline_rate is not None else 0.5
    base = clamp(base, 0.02, 0.98)

    n = normalize_scores(cgpa, gre, ielts, lor_sop)

    # weights: GPA 0.4, GRE 0.35, IELTS 0.15, LOR/SOP 0.10 (missing values drop and renormalize)
    feats = []
    weights = []
    if n["gpa"] is not None:
        feats.append(n["gpa"]); weights.append(0.4)
    if n["gre"] is not None:
        feats.append(n["gre"]); weights.append(0.35)
    if n["ielts"] is not None:
        feats.append(n["ielts"]); weights.append(0.15)
    if n["lor"] is not None:
        feats.append(n["lor"]); weights.append(0.10)
    if not feats:
        return base
    wsum = sum(weights)
    match = sum(f*w for f, w in zip(feats, weights))/wsum

    # logit shift around baseline; beta controls sensitivity
    beta = 1.6
    z = logit(base) + beta*(match - 0.5)
    return float(clamp(inv_logit(z), 0.01, 0.99))
