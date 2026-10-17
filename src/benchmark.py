"""Label-free query-count scaling benchmark on the first training cutoff."""
import argparse
import time

import numpy as np
import pandas as pd

from .experiment import Experiment
from .features import history_before,eligible_users,candidates,build_features


def main():
    exp=Experiment()
    cutoff=pd.Timestamp(exp.manifest['cutoffs']['train_1'])
    history=history_before(exp.events,cutoff)
    users=eligible_users(history,cutoff)
    order=np.random.default_rng(43).permutation(users)
    records=[]
    for n in [100,500,1000]:
        chosen=np.sort(order[:n])
        cand,ct=candidates(history,exp.items,chosen,cutoff)
        X,manual,_,ft=build_features(history,exp.items,cand,cutoff)
        records.append(dict(n_users=n,n_pairs=len(cand),manual_bytes=int(X[manual].memory_usage(deep=True).sum()),**ct,**ft))
        print(records[-1],flush=True)
    pd.DataFrame(records).to_csv(exp.out/'scaling.csv',index=False)


if __name__=='__main__': main()
