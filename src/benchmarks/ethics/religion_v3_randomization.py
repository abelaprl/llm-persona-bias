"""Paired label-exchange tests, exact for at most 16 informative pairs."""
from functools import lru_cache
import numpy as np


@lru_cache(maxsize=17)
def exact_signs(n):
    values=np.arange(2**n,dtype=np.uint32)[:,None]
    return (((values>>np.arange(n,dtype=np.uint32))&1).astype(float)*2-1)


def paired_sign_flip(differences,weights,seed,resamples=10000):
    differences=np.asarray(differences,float)
    weights=np.asarray(weights,float)
    if differences.ndim!=2 or weights.shape!=(len(differences),):raise ValueError('Invalid paired arrays')
    if not np.isfinite(weights).all() or (weights<0).any():raise ValueError('Invalid weights')
    columns=differences.shape[1]
    finite=np.isfinite(differences).all(axis=0)
    informative=(np.abs(differences)>1e-12).sum(axis=0)
    p=np.full(columns,np.nan)
    methods=np.full(columns,'not_estimated_insufficient_support',dtype=object)
    if len(differences)<2:return p,methods,informative
    weighted=np.where(np.abs(differences)>1e-12,differences,0.)*weights[:,None]
    observed=np.abs(weighted.sum(axis=0))
    tolerance=np.maximum(1.,np.abs(weighted).sum(axis=0))*1e-12
    for col in np.flatnonzero(finite&(informative<=16)):
        values=weighted[:,col][np.abs(differences[:,col])>1e-12]
        simulated=exact_signs(len(values))@values
        p[col]=np.mean(np.abs(simulated)>=observed[col]-tolerance[col])
        methods[col]='exact_paired_sign_flip'
    large=np.flatnonzero(finite&(informative>16))
    if len(large):
        values=weighted[:,large]
        active=np.any(values!=0,axis=1)
        values=values[active]
        rng=np.random.default_rng(seed)
        exceed=np.zeros(len(large),dtype=np.int64)
        for start in range(0,resamples,256):
            batch=min(256,resamples-start)
            signs=rng.integers(0,2,size=(batch,len(values)),dtype=np.int8).astype(float)*2-1
            simulated=signs@values
            exceed+=(np.abs(simulated)>=observed[large]-tolerance[large]).sum(axis=0)
        p[large]=(exceed+1)/(resamples+1)
        methods[large]='monte_carlo_paired_sign_flip'
    return p,methods,informative
