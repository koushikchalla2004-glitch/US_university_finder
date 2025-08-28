import math

USD = "${:,.0f}"

def safe(val, default=None):
    return default if val in (None, "", float("nan")) else val

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def logit(p):
    p = clamp(p, 1e-6, 1-1e-6)
    return math.log(p/(1-p))

def inv_logit(z):
    return 1/(1+math.exp(-z))
