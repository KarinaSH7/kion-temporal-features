"""Explicit stages enforce test isolation: build-dev -> select -> final-test."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from .data import ROOT, dump, sha256
from .features import CAT_COLS, RELATIONS, history_before, eligible_users, candidates, build_features, attach_labels
from .metrics import evaluate, paired_bootstrap


class Experiment:
    def __init__(self, n_users=10000):
        self.data=ROOT/f'data/cohort_{n_users}'
        self.out=ROOT/('results/main' if n_users==10000 else f'results/users_{n_users}')
        self.cache=self.out/'cache'
        self.models=self.out/'models'
        for path in [self.out,self.cache,self.models,self.out/'predictions']:
            path.mkdir(parents=True,exist_ok=True)
        self.manifest=json.loads((self.data/'manifest.json').read_text())
        self.events=pd.read_parquet(self.data/'events.parquet')
        self.items=pd.read_parquet(self.data/'items.parquet')

    def build(self, name):
        start=time.perf_counter()
        cutoff=pd.Timestamp(self.manifest['cutoffs'][name])
        h=history_before(self.events,cutoff)
        users=eligible_users(h,cutoff,2000 if name.startswith('train') else None)
        cand,timing=candidates(h,self.items,users,cutoff)
        X,manual,specs,ft=build_features(h,self.items,cand,cutoff)
        # Candidate rows and X exist before future events can influence the stage.
        fingerprint=hashlib.sha256(pd.util.hash_pandas_object(cand[['user_id','item_id']],index=False).values.tobytes()).hexdigest()
        y,relevance,coverage=attach_labels(self.events,h,cand,users,cutoff)
        cand['label']=y
        X.to_parquet(self.cache/f'{name}_X.parquet',index=False)
        cand.to_parquet(self.cache/f'{name}_candidates.parquet',index=False)
        relevance.rename('n_relevant').to_frame().to_parquet(self.cache/f'{name}_relevance.parquet')
        timing.update(ft)
        timing.update(cutoff=str(cutoff),end=str(cutoff+pd.Timedelta(days=7)),
                      positive_queries=int((relevance>0).sum()),positives=int(relevance.sum()),
                      candidate_positives=int(y.sum()),known_catalog_target_coverage=coverage,
                      candidates_sha256=fingerprint,total_seconds=time.perf_counter()-start)
        dump(self.out/f'{name}_statistics.json',timing)
        if name=='train_1':
            dump(self.out/'feature_schema.json',dict(manual=manual,all=list(X.columns),grammar=specs))
        print(f'BUILT {name}: {json.dumps(timing)}',flush=True)

    def load(self,name):
        X=pd.read_parquet(self.cache/f'{name}_X.parquet')
        cand=pd.read_parquet(self.cache/f'{name}_candidates.parquet')
        rel=pd.read_parquet(self.cache/f'{name}_relevance.parquet').n_relevant
        return X,cand,cand.label.to_numpy(),rel

    def training(self):
        parts=[self.load(n) for n in ['train_1','train_2','train_3']]
        return pd.concat([p[0] for p in parts],ignore_index=True), np.concatenate([p[2] for p in parts])

    @staticmethod
    def pool(X,y=None):
        return Pool(X,label=y,cat_features=[c for c in CAT_COLS if c in X.columns])

    @staticmethod
    def model(depth=5,iterations=300,seed=42):
        return CatBoostClassifier(iterations=iterations,depth=depth,learning_rate=.07,l2_leaf_reg=5,
                                  loss_function='Logloss',random_seed=seed,thread_count=4,
                                  verbose=False,allow_writing_files=False,one_hot_max_size=20)

    def select(self):
        if (self.out/'test_metrics.csv').exists():
            raise RuntimeError('Test was already opened. Use a new results directory for a new experiment.')
        X,y=self.training()
        SX,SC,SY,SR=self.load('search')
        TX,TC,TY,TR=self.load('tune')
        schema=json.loads((self.out/'feature_schema.json').read_text())
        manual=schema['manual']
        allcols=schema['all']
        start=time.perf_counter()
        pilot=self.model(iterations=200)
        pilot.fit(self.pool(X,y))
        importance=pd.DataFrame({'feature':allcols,'importance':pilot.feature_importances_}).sort_values(['importance','feature'],ascending=[False,True])
        importance.to_csv(self.out/'pilot_importance.csv',index=False)
        extra=[f for f in importance.feature if f not in manual and float(importance.loc[importance.feature==f,'importance'].iloc[0])>0]
        options=[]
        for size in [12,24,48]:
            cols=manual+extra[:size]
            model=self.model(iterations=200)
            t=time.perf_counter(); model.fit(self.pool(X[cols],y)); duration=time.perf_counter()-t
            scores=model.predict_proba(self.pool(SX[cols]))[:,1]
            metrics,_=evaluate(SC,SY,scores,SR)
            options.append(dict(size=size,n_features=len(cols),train_seconds=duration,**metrics))
            print(f'SEARCH top={size} NDCG={metrics["ndcg10"]:.6f}',flush=True)
        winner=max(options,key=lambda a:(a['ndcg10'],-a['size']))
        selected=manual+extra[:winner['size']]
        search_time=time.perf_counter()-start
        pd.DataFrame(options).to_csv(self.out/'search_trials.csv',index=False)
        # Remove all cross-history/metadata and collaborative scores, but keep fixed retrieval.
        indirect=lambda c: any(c.startswith('u_'+r+'_') for r in RELATIONS) or c in ['cf_score','genre_cosine','retrieval_score','retrieval_rank']
        arms={'manual':manual,'auto_all':allcols,'auto_selected':selected,
              'no_indirect':[c for c in manual if not indirect(c)]}
        trials=[]; chosen={}
        for arm,cols in arms.items():
            best=None
            for depth in [5,7]:
                model=self.model(depth=depth)
                t=time.perf_counter(); model.fit(self.pool(X[cols],y)); duration=time.perf_counter()-t
                pool=self.pool(TX[cols])
                for count in [100,200,300]:
                    pred=model.predict_proba(pool,ntree_end=count)[:,1]
                    metrics,_=evaluate(TC,TY,pred,TR)
                    rec=dict(arm=arm,depth=depth,iterations=count,n_features=len(cols),full_fit_seconds=duration,**metrics)
                    trials.append(rec)
                    if best is None or rec['ndcg10']>best['ndcg10']:
                        best=rec.copy()
                        model.save_model(str(self.models/f'{arm}_seed42.cbm'))
                print(f'TUNE {arm} depth={depth} best={best["ndcg10"]:.6f}',flush=True)
            # Saved model may contain 300 trees; slice to selected checkpoint only after loading.
            saved=CatBoostClassifier(); saved.load_model(str(self.models/f'{arm}_seed42.cbm'))
            saved.shrink(best['iterations']); saved.save_model(str(self.models/f'{arm}_seed42.cbm'))
            chosen[arm]=dict(columns=cols,depth=best['depth'],iterations=best['iterations'],validation=best)
        pd.DataFrame(trials).to_csv(self.out/'tuning_trials.csv',index=False)
        lock=dict(protocol_sha256=sha256(ROOT/'PROTOCOL.md'),cohort_sha256=self.manifest['cohort_sha256'],
                  chosen=chosen,search_seconds=search_time,search_top_k=winner['size'],
                  extra_feature_ranking=extra,created_at_utc=pd.Timestamp.now(tz='UTC').isoformat(),
                  primary_comparison='auto_selected minus manual',seeds=[42,17,2026],
                  test_used_for_selection=False)
        dump(self.out/'locked_config.json',lock)
        dump(self.out/'selection_environment.json',dict(python=platform.python_version(),platform=platform.platform(),
                    cpu_count=os.cpu_count(),catboost_threads=4,peak_rss_raw=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        print('CONFIGURATION LOCKED; test not loaded',flush=True)

    def final_test(self):
        lock=json.loads((self.out/'locked_config.json').read_text())
        assert lock['protocol_sha256']==sha256(ROOT/'PROTOCOL.md')
        if (self.out/'test_metrics.csv').exists():
            raise RuntimeError('Results already exist; refusing accidental repeated test tuning.')
        self.build('test')
        X,C,Y,R=self.load('test')
        oracle,_=evaluate(C,Y,Y,R,probabilities=False)
        dump(self.out/'candidate_oracle.json',dict(description='Diagnostic ceiling: perfect ordering of the FIXED retrieved candidates, not a trained model',**oracle))
        train,train_y=self.training()
        records=[]; per={}; times=[]
        for arm,spec in lock['chosen'].items():
            for seed in ([42,17,2026] if arm in ['manual','auto_selected'] else [42]):
                cols=spec['columns']; t=time.perf_counter()
                if seed==42:
                    model=CatBoostClassifier(); model.load_model(str(self.models/f'{arm}_seed42.cbm'))
                    fit_seconds=None
                else:
                    model=self.model(depth=spec['depth'],iterations=spec['iterations'],seed=seed)
                    model.fit(self.pool(train[cols],train_y))
                    fit_seconds=time.perf_counter()-t
                    model.save_model(str(self.models/f'{arm}_seed{seed}.cbm'))
                t=time.perf_counter(); pool=self.pool(X[cols]); pool_seconds=time.perf_counter()-t
                t=time.perf_counter(); pred=model.predict_proba(pool)[:,1]; infer_seconds=time.perf_counter()-t
                metrics,pu=evaluate(C,Y,pred,R)
                name=f'{arm}_seed{seed}'; per[name]=pu
                pu.to_csv(self.out/'predictions'/f'{name}_per_user.csv')
                np.save(self.out/'predictions'/f'{name}.npy',pred)
                records.append(dict(arm=arm,seed=seed,n_features=len(cols),depth=spec['depth'],iterations=spec['iterations'],**metrics))
                times.append(dict(arm=arm,seed=seed,fit_seconds=fit_seconds,pool_seconds=pool_seconds,
                                  inference_seconds=infer_seconds,model_bytes=(self.models/f'{name}.cbm').stat().st_size))
                if seed==42:
                    values=model.get_feature_importance(self.pool(train[cols],train_y),type='PredictionValuesChange')
                    assert np.isfinite(values).all()
                    pd.DataFrame({'feature':cols,'importance':values}).sort_values('importance',ascending=False).to_csv(self.out/f'{arm}_importance.csv',index=False)
                print(f'TEST {name}: {json.dumps(metrics)}',flush=True)
        # History-only popularity baseline, same candidates.
        scores=X.i_count_7.to_numpy()+X.i_count_all.to_numpy()*1e-6
        metrics,pu=evaluate(C,Y,scores,R,probabilities=False)
        per['popularity']=pu
        pu.to_csv(self.out/'predictions'/'popularity_per_user.csv')
        records.append(dict(arm='popularity',seed=42,n_features=2,**metrics))
        pd.DataFrame(records).to_csv(self.out/'test_metrics.csv',index=False)
        pd.DataFrame(times).to_csv(self.out/'runtime.csv',index=False)
        ci={str(seed):paired_bootstrap(per[f'manual_seed{seed}'],per[f'auto_selected_seed{seed}']) for seed in [42,17,2026]}
        dump(self.out/'paired_bootstrap.json',ci)
        # Fixed stratification thresholds, descriptive analysis only.
        history_counts=C[['user_id']].assign(count=X.u_count_all).groupby('user_id')['count'].first()
        strata=[]
        for name,pu in per.items():
            q=pu.join(history_counts)
            q['stratum']=pd.cut(q['count'],[-1,2,5,20,np.inf],labels=['1-2','3-5','6-20','21+'])
            for group,part in q[q.n_relevant>0].groupby('stratum',observed=True):
                strata.append(dict(model=name,stratum=str(group),users=len(part),ndcg10=part.ndcg.mean(),recall10=part.recall.mean()))
        pd.DataFrame(strata).to_csv(self.out/'activity_strata.csv',index=False)
        dump(self.out/'test_audit.json',dict(locked_config_sha256=sha256(self.out/'locked_config.json'),
                 peak_rss_raw=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                 finished_at_utc=pd.Timestamp.now(tz='UTC').isoformat(),source_checksums=self.manifest['checksums']))

    def explain(self):
        """Export training-based explanations of existing models; no fitting/test tuning."""
        lock=json.loads((self.out/'locked_config.json').read_text())
        train,y=self.training()
        for arm,spec in lock['chosen'].items():
            model=CatBoostClassifier(); model.load_model(str(self.models/f'{arm}_seed42.cbm'))
            values=model.get_feature_importance(self.pool(train[spec['columns']],y),type='PredictionValuesChange')
            assert np.isfinite(values).all() and abs(values.sum()-100)<1e-5
            pd.DataFrame({'feature':spec['columns'],'importance':values}).sort_values('importance',ascending=False).to_csv(self.out/f'{arm}_importance.csv',index=False)
        print('Exported finite training-based importances; models and predictions unchanged',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['build-dev','select','final-test','explain'])
    parser.add_argument('--users',type=int,default=10000)
    args=parser.parse_args()
    exp=Experiment(args.users)
    if args.stage=='build-dev':
        for name in ['train_1','train_2','train_3','search','tune']: exp.build(name)
    elif args.stage=='select': exp.select()
    elif args.stage=='explain': exp.explain()
    else: exp.final_test()
