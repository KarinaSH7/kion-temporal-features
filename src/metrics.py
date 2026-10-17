"""End-to-end metrics: positives missed by retrieval remain in denominators."""
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score


def per_user(cand, y, scores, relevance, k=10):
    ranked=cand[['user_id','item_id']].copy()
    ranked['label']=y
    ranked['score']=scores
    ranked=ranked.sort_values(['user_id','score','item_id'],ascending=[True,False,True])
    ranked['rank']=ranked.groupby('user_id').cumcount()+1
    top=ranked[ranked['rank']<=k].copy()
    top['gain']=top.label/np.log2(top['rank']+1)
    out=pd.DataFrame({'n_relevant':relevance})
    out['hits']=top.groupby('user_id').label.sum().reindex(out.index,fill_value=0)
    out['retrieved']=ranked.groupby('user_id').label.sum().reindex(out.index,fill_value=0)
    ideal=np.concatenate([[0],np.cumsum(1/np.log2(np.arange(1,k+1)+1))])
    out['ndcg']=top.groupby('user_id').gain.sum().reindex(out.index,fill_value=0).to_numpy()/np.maximum(ideal[np.minimum(out.n_relevant.to_numpy(),k)],1e-12)
    out['recall']=out.hits/np.maximum(out.n_relevant,1)
    out['candidate_recall']=out.retrieved/np.maximum(out.n_relevant,1)
    return out


def evaluate(cand,y,scores,relevance,probabilities=True):
    per=per_user(cand,y,scores,relevance)
    active=per[per.n_relevant>0]
    metrics=dict(ndcg10=float(active.ndcg.mean()),recall10=float(active.recall.mean()),
                 ndcg10_all=float(per.ndcg.mean()),candidate_recall100=float(active.candidate_recall.mean()),
                 queries=len(per),positive_queries=len(active),candidate_positives=int(np.sum(y)),
                 positive_rate=float(np.mean(y)),auc=float(roc_auc_score(y,scores)) if len(np.unique(y))==2 else None)
    if probabilities:
        metrics['logloss']=float(log_loss(y,np.clip(scores,1e-8,1-1e-8),labels=[0,1]))
    return metrics,per


def paired_bootstrap(a,b,seed=42,n=2000):
    merged=a[['ndcg','n_relevant']].join(b[['ndcg']],lsuffix='_a',rsuffix='_b',validate='one_to_one')
    delta=(merged.ndcg_b-merged.ndcg_a)[merged.n_relevant>0].to_numpy()
    rng=np.random.default_rng(seed)
    means=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(n)])
    return dict(delta=float(delta.mean()),ci95_low=float(np.quantile(means,.025)),ci95_high=float(np.quantile(means,.975)),
                n_users=len(delta),bootstrap_replicates=n,seed=seed)
