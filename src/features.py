"""Pure point-in-time feature/candidate functions: labels are a separate operation."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

RELATIONS = ["genre", "country", "director", "content_type"]
MANUAL_WINDOWS = [7, 30, None]
AUTO_WINDOWS = [1, 3, 10, 14, 21, 60]
CAT_COLS = ["genre", "country", "director", "content_type"]


def metadata(items):
    m = items.set_index("item_id").copy()
    for new, old in [("genre", "genres"), ("country", "countries"), ("director", "directors")]:
        m[new] = m[old].fillna("__missing__").str.split(",").str[0].str.strip()
    m["content_type"] = m.content_type.fillna("__missing__")
    for key in ["release_year", "age_rating", "for_kids"]:
        m[key] = pd.to_numeric(m[key], errors="coerce").fillna(-1)
    return m


def history_before(events, cutoff):
    return events.loc[events.timestamp < pd.Timestamp(cutoff)].copy()


def eligible_users(history, cutoff, max_users=None, seed=42):
    users = np.sort(history.loc[history.timestamp >= pd.Timestamp(cutoff)-pd.Timedelta(days=28), "user_id"].unique())
    if max_users is not None and len(users) > max_users:
        users = np.sort(np.random.default_rng(seed).choice(users, max_users, replace=False))
    return users


def candidates(history, items, users, cutoff, n=100):
    """History-only fusion. No positive injection or future-item catalogue."""
    start = time.perf_counter()
    cutoff = pd.Timestamp(cutoff)
    if len(history) == 0:
        raise ValueError("Nonempty history is required")
    assert (history.timestamp < cutoff).all()
    item_ids = np.sort(history.item_id.unique())
    user_ids = np.sort(history.user_id.unique())
    ui = pd.Index(user_ids).get_indexer(history.user_id)
    ii = pd.Index(item_ids).get_indexer(history.item_id)
    mat = sparse.csr_matrix((np.ones(len(history), dtype=np.float32), (ui, ii)), shape=(len(user_ids), len(item_ids)))
    mat.data[:] = 1
    popularity = np.asarray(mat.sum(axis=0)).ravel()
    inv = 1 / np.sqrt(np.maximum(popularity, 1))
    # Cosine with a fixed shrinkage term to suppress one-off coincidences.
    co = (mat.T @ mat).tocoo()
    counts = co.data.copy()
    co.data *= inv[co.row] * inv[co.col] * counts / (counts + 10)
    sim = co.tocsr()
    sim.setdiag(0)
    sim.eliminate_zeros()
    # Prune each item's neighbourhood to at most 50, deterministic ties by item ID.
    for r in range(sim.shape[0]):
        a,b = sim.indptr[r:r+2]
        if b-a > 50:
            order = np.lexsort((sim.indices[a:b], -sim.data[a:b]))
            sim.data[a+order[50:]] = 0
    sim.eliminate_zeros()
    m = metadata(items).reindex(item_ids)
    genres = m.genres.fillna("__missing__").str.split(",").apply(lambda v: sorted(set(s.strip() for s in v)))
    vocabulary = sorted({g for gs in genres for g in gs})
    gi = {g:k for k,g in enumerate(vocabulary)}
    gr, gc = [], []
    for i,gs in enumerate(genres):
        gr.extend([i]*len(gs)); gc.extend([gi[g] for g in gs])
    genre_mat = sparse.csr_matrix((np.ones(len(gr), dtype=np.float32), (gr,gc)), shape=(len(item_ids),len(gi)))
    genre_mat = sparse.diags(1 / np.sqrt(np.maximum(np.asarray(genre_mat.sum(axis=1)).ravel(), 1))) @ genre_mat
    recent = history[history.timestamp >= cutoff-pd.Timedelta(days=7)].groupby('item_id').size().reindex(item_ids, fill_value=0).to_numpy()
    trend_order = np.lexsort((item_ids, -recent))
    all_order = np.lexsort((item_ids, -popularity))
    user_index = pd.Index(user_ids)
    rows = []
    for user in users:
        row = mat.getrow(user_index.get_loc(user))
        seen = set(row.indices)
        cf = (row @ sim).toarray().ravel()
        affinity = (row @ genre_mat @ genre_mat.T).toarray().ravel()
        affinity /= max(float(row.sum()), 1)
        # Genre channel retains popularity information to avoid obscure equal-score items.
        genre_score = affinity * np.log1p(popularity)
        orders = [trend_order, all_order, np.lexsort((item_ids, -cf)), np.lexsort((item_ids, -genre_score))]
        channel_scores = [recent, popularity, cf, genre_score]
        rank_maps = []
        pool = set()
        for order, scores in zip(orders, channel_scores):
            eligible = [int(i) for i in order if i not in seen and scores[i] > 0][:n]
            rank_maps.append({i:r+1 for r,i in enumerate(eligible)})
            pool.update(eligible)
        rrf = {i: sum(1/(60+rm[i]) for rm in rank_maps if i in rm) for i in pool}
        chosen = sorted(pool, key=lambda i: (-rrf[i], item_ids[i]))[:n]
        for rank,i in enumerate(chosen):
            rows.append((int(user), int(item_ids[i]), float(cf[i]), float(affinity[i]), float(rrf[i]), rank+1))
    frame = pd.DataFrame(rows, columns=['user_id','item_id','cf_score','genre_cosine','retrieval_score','retrieval_rank'])
    assert not frame.duplicated(['user_id','item_id']).any()
    observed = pd.MultiIndex.from_frame(history[['user_id','item_id']])
    assert not pd.MultiIndex.from_frame(frame[['user_id','item_id']]).isin(observed).any()
    return frame, dict(candidate_seconds=time.perf_counter()-start, known_items=len(item_ids),
                       history_rows=len(history), queries=len(users), candidate_rows=len(frame))


def _lookup(series, rows, keys, default=0):
    index = rows[keys[0]] if len(keys)==1 else pd.MultiIndex.from_frame(rows[keys])
    return series.reindex(index).fillna(default).to_numpy(dtype=np.float32)


def build_features(history, items, cand, cutoff):
    start = time.perf_counter()
    cutoff = pd.Timestamp(cutoff)
    assert (history.timestamp < cutoff).all()
    meta = metadata(items)
    h = history.merge(meta[RELATIONS], left_on='item_id', right_index=True, how='left', validate='many_to_one')
    q = cand.merge(meta[RELATIONS+['release_year','age_rating','for_kids']], left_on='item_id', right_index=True, how='left', validate='many_to_one')
    q[RELATIONS] = q[RELATIONS].fillna('__missing__')
    h[RELATIONS] = h[RELATIONS].fillna('__missing__')
    data = {c:q[c].to_numpy() for c in ['cf_score','genre_cosine','retrieval_score','retrieval_rank']+RELATIONS+['release_year','age_rating','for_kids']}
    data['item_age_years'] = cutoff.year - q.release_year.to_numpy()
    groups = { 'u':['user_id'], 'i':['item_id'], 'ui':['user_id','item_id'] }
    groups.update({f'u_{r}':['user_id',r] for r in RELATIONS})
    for name,keys in groups.items():
        last = h.groupby(keys).timestamp.max()
        age = (cutoff-last).dt.total_seconds()/86400
        data[f'{name}_recency'] = _lookup(age,q,keys,default=3650)
    data['u_unique_items'] = _lookup(h.groupby('user_id').item_id.nunique(),q,['user_id'])
    data['u_active_days'] = _lookup(h.groupby('user_id').timestamp.nunique(),q,['user_id'])
    data['u_unique_genres'] = _lookup(h.groupby('user_id').genre.nunique(),q,['user_id'])
    manual = list(data)
    group_specs = []
    for w in MANUAL_WINDOWS + AUTO_WINDOWS:
        label = 'all' if w is None else str(w)
        part = h if w is None else h[h.timestamp >= cutoff-pd.Timedelta(days=w)]
        group_columns = []
        for name,keys in groups.items():
            col = f'{name}_count_{label}'
            data[col] = _lookup(part.groupby(keys).size(),q,keys)
            group_columns.append(col)
        for r in RELATIONS:
            col=f'u_{r}_share_{label}'
            data[col] = data[f'u_{r}_count_{label}'] / (data[f'u_count_{label}']+1)
            group_columns.append(col)
        col=f'i_share_{label}'
        data[col] = data[f'i_count_{label}'] / max(len(part),1)
        group_columns.append(col)
        if w in MANUAL_WINDOWS:
            manual.extend(group_columns)
        group_specs.append(dict(operator='window_count_and_share', window_days=w, columns=group_columns))
    for name in ['u','i']+[f'u_{r}' for r in RELATIONS]:
        col=f'{name}_trend_7_30'
        data[col]=(data[f'{name}_count_7']+1)/(data[f'{name}_count_30']+4)
        manual.append(col)
    for half in [3,10,30]:
        tmp = h.copy()
        tmp['_weight'] = np.exp2(-(cutoff-h.timestamp).dt.total_seconds()/(86400*half))
        cols=[]
        for name,keys in groups.items():
            col=f'{name}_decay_{half}'
            data[col]=_lookup(tmp.groupby(keys)._weight.sum(),q,keys)
            cols.append(col)
        group_specs.append(dict(operator='exponential_decay',half_life_days=half,columns=cols))
    for w in AUTO_WINDOWS:
        for name in ['u','i']+[f'u_{r}' for r in RELATIONS]:
            data[f'{name}_rate_ratio_{w}_30']=(data[f'{name}_count_{w}']+1)/(data[f'{name}_count_30']+30/w)
    frame = pd.DataFrame(data)
    for c in frame.columns:
        if c not in CAT_COLS:
            frame[c]=pd.to_numeric(frame[c],errors='raise').fillna(-1).astype('float32')
    assert np.isfinite(frame.drop(columns=CAT_COLS).to_numpy()).all()
    return frame, manual, group_specs, dict(feature_seconds=time.perf_counter()-start,
                                           feature_bytes=int(frame.memory_usage(deep=True).sum()),
                                           feature_count=len(frame.columns),manual_count=len(manual))


def attach_labels(events, history, cand, users, cutoff, horizon=7):
    cutoff=pd.Timestamp(cutoff)
    future=events[(events.timestamp>=cutoff)&(events.timestamp<cutoff+pd.Timedelta(days=horizon))&events.user_id.isin(users)]
    old=pd.MultiIndex.from_frame(history[['user_id','item_id']])
    future=future[~pd.MultiIndex.from_frame(future[['user_id','item_id']]).isin(old)]
    target=pd.MultiIndex.from_frame(future[['user_id','item_id']])
    y=pd.MultiIndex.from_frame(cand[['user_id','item_id']]).isin(target).astype(np.int8)
    relevance=future.groupby('user_id').item_id.nunique().reindex(users,fill_value=0)
    known=set(history.item_id)
    coverage=future.item_id.isin(known).sum()/max(len(future),1)
    return y, relevance, float(coverage)
