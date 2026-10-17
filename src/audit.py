"""Independent brute-force checks against raw sampled events and saved artifacts."""
import argparse
import json

import numpy as np
import pandas as pd

from .data import ROOT,dump
from .features import metadata
from .experiment import Experiment


def main(include_test=False):
    exp=Experiment()
    rng=np.random.default_rng(901)
    checks=[]
    names=['train_1','train_2','train_3','search','tune']+(['test'] if include_test else [])
    for name in names:
        X,C,Y,R=exp.load(name)
        cutoff=pd.Timestamp(exp.manifest['cutoffs'][name])
        h=exp.events[exp.events.timestamp<cutoff]
        h=h.merge(metadata(exp.items)[['genre']],left_on='item_id',right_index=True,how='left')
        m=metadata(exp.items)
        assert not C.duplicated(['user_id','item_id']).any()
        assert len(C.merge(h,on=['user_id','item_id']))==0
        assert set(C.item_id).issubset(set(h.item_id))
        assert C.groupby('user_id').size().eq(100).all()
        for row in rng.choice(len(C),min(100,len(C)),replace=False):
            user,item=C.iloc[row][['user_id','item_id']]
            user_history=h[h.user_id==user]
            item_history=h[h.item_id==item]
            for window in [1,7,10,30,60]:
                left=cutoff-pd.Timedelta(days=window)
                assert X.iloc[row][f'u_count_{window}']==int((user_history.timestamp>=left).sum())
                assert X.iloc[row][f'i_count_{window}']==int((item_history.timestamp>=left).sum())
            expected=len(user_history[user_history.genre==m.loc[item,'genre']])
            assert X.iloc[row].u_genre_count_all==expected
            target=exp.events[(exp.events.user_id==user)&(exp.events.item_id==item)&
                    (exp.events.timestamp>=cutoff)&(exp.events.timestamp<cutoff+pd.Timedelta(days=7))]
            assert Y[row]==int(len(target)>0)
        checks.append(dict(split=name,random_rows_checked=100,assertions_per_row=12,
                           candidate_rows=len(C),unique_pairs=True,no_seen_pairs=True,known_catalog_only=True))
    # Re-evaluate exported test predictions with a second, explicit loop implementation.
    if include_test:
        X,C,Y,R=exp.load('test')
        expected=pd.read_csv(exp.out/'test_metrics.csv')
        for arm in ['manual','auto_selected']:
            scores=np.load(exp.out/'predictions'/f'{arm}_seed42.npy')
            z=C[['user_id','item_id','label']].assign(score=scores)
            dcgs=[];recalls=[]
            for user,group in z.groupby('user_id'):
                if R.loc[user]==0: continue
                top=group.sort_values(['score','item_id'],ascending=[False,True]).head(10)
                dcg=sum(float(v)/np.log2(rank+2) for rank,v in enumerate(top.label))
                ideal=sum(1/np.log2(j+2) for j in range(min(10,int(R.loc[user]))))
                dcgs.append(dcg/ideal);recalls.append(top.label.sum()/R.loc[user])
            record=expected[(expected.arm==arm)&(expected.seed==42)].iloc[0]
            assert abs(np.mean(dcgs)-record.ndcg10)<1e-12
            assert abs(np.mean(recalls)-record.recall10)<1e-12
    dump(exp.out/('audit_final.json' if include_test else 'audit_development.json'),
         dict(status='passed',seed=901,splits=checks,independent_test_metrics_checked=include_test))
    print('Independent audit passed:',len(checks),'splits',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--include-test',action='store_true')
    main(parser.parse_args().include_test)
