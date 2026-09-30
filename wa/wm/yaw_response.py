"""Bounded controller ablation, NOT a unit correction or adopted policy."""
import math
def yaw_gain2(action):
    a=list(action)
    if len(a)!=3 or not all(math.isfinite(float(x)) for x in a):
        raise ValueError('invalid action')
    if any(abs(float(x))>1 for x in a):raise ValueError('expected normalized action')
    a[2]=max(-1.,min(1.,2.*a[2]))
    return a
