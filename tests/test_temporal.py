import numpy as np
import pandas as pd
import pytest

from src.data import sample_users
from src.features import history_before, eligible_users, candidates, build_features, attach_labels
from src.metrics import evaluate, paired_bootstrap


@pytest.fixture
def fixture():
    events=pd.DataFrame([
        (1,10,'2021-01-01'),(1,11,'2021-01-03'),(2,12,'2021-01-02'),
        (2,13,'2021-01-04'),(3,14,'2021-01-02'),(3,15,'2021-01-04'),
        (1,12,'2021-01-08'),(1,99,'2021-01-09'),(4,16,'2021-01-09')],
        columns=['user_id','item_id','timestamp'])
    events.timestamp=pd.to_datetime(events.timestamp)
    ids=[10,11,12,13,14,15,16,99]
    items=pd.DataFrame(dict(item_id=ids,genres=['a, b']*8,countries=['x']*8,
                            directors=['d']*8,content_type=['film']*8,
                            release_year=[2020]*8,age_rating=[16]*8,for_kids=[0]*8))
    return events,items,pd.Timestamp('2021-01-08')


def test_history_strict_boundary(fixture):
    events,_,cut=fixture
    h=history_before(events,cut)
    assert len(h)==6
    assert h.timestamp.max()<cut
    assert 4 not in eligible_users(h,cut)


def test_future_mutation_cannot_change_features_or_candidates(fixture):
    events,items,cut=fixture
    mutated=pd.concat([events, pd.DataFrame({'user_id':[1,77],'item_id':[99,99],
                         'timestamp':[cut,cut+pd.Timedelta(days=1)]})],ignore_index=True)
    outputs=[]
    for data in [events,mutated]:
        h=history_before(data,cut)
        users=eligible_users(h,cut)
        c,_=candidates(h,items,users,cut,n=3)
        X,*_=build_features(h,items,c,cut)
        outputs.append((c,X))
    pd.testing.assert_frame_equal(outputs[0][0],outputs[1][0])
    pd.testing.assert_frame_equal(outputs[0][1],outputs[1][1])


def test_no_future_item_or_seen_pair_in_retrieval(fixture):
    events,items,cut=fixture
    h=history_before(events,cut)
    users=eligible_users(h,cut)
    c,_=candidates(h,items,users,cut,n=100)
    assert not set(c.item_id)&{16,99}
    assert len(c.merge(h,on=['user_id','item_id']))==0
    y,rel,coverage=attach_labels(events,h,c,users,cut)
    assert rel.loc[1]==2
    assert y.sum()==1  # item 99 remains missed, never injected.
    assert coverage==.5


def test_window_left_closed_and_right_open(fixture):
    events,items,cut=fixture
    h=history_before(events,cut)
    c=pd.DataFrame(dict(user_id=[1],item_id=[12],cf_score=[0.],genre_cosine=[0.],retrieval_score=[0.],retrieval_rank=[1]))
    X,manual,_,_=build_features(h,items,c,cut)
    assert X.u_count_7.iloc[0]==2  # Jan 1 is included at Jan 8 cutoff.
    assert X.u_count_3.iloc[0]==0
    assert X.ui_count_all.iloc[0]==0
    assert X.u_recency.iloc[0]==5
    assert X.u_genre_count_all.iloc[0]==2
    assert 'u_count_7' in manual and 'u_count_10' not in manual
    assert X.u_decay_3.iloc[0]==pytest.approx(2**(-7/3)+2**(-5/3))


def test_no_history_guard(fixture):
    events,items,cut=fixture
    with pytest.raises(AssertionError):
        build_features(events,items,pd.DataFrame(),cut)


def test_complete_user_sampling(fixture):
    events,_,cut=fixture
    sample,ids=sample_users(events,2,42,cut)
    assert 4 not in ids  # user not yet known at the first cutoff
    pd.testing.assert_series_equal(events[events.user_id.isin(ids)].groupby('user_id').size(),sample.groupby('user_id').size())
    again,ids_again=sample_users(events,2,42,cut)
    np.testing.assert_array_equal(ids,ids_again)
    pd.testing.assert_frame_equal(sample,again)


def test_ndcg_denominator_contains_unretrieved_positives():
    cand=pd.DataFrame({'user_id':[1,1,2],'item_id':[10,11,12]})
    rel=pd.Series([2,0],index=pd.Index([1,2],name='user_id'))
    metrics,per=evaluate(cand,np.array([1,0,0]),np.array([.9,.2,.1]),rel)
    expected=1/(1+1/np.log2(3))
    assert metrics['ndcg10']==pytest.approx(expected)
    assert metrics['ndcg10_all']==pytest.approx(expected/2)
    assert metrics['recall10']==.5
    assert metrics['candidate_recall100']==.5
    assert per.loc[2,'ndcg']==0


def test_ties_are_deterministic():
    c=pd.DataFrame({'user_id':[1,1],'item_id':[2,1]})
    r=pd.Series([1],index=pd.Index([1],name='user_id'))
    a,p=evaluate(c,[0,1],[.5,.5],r)
    b,q=evaluate(c.iloc[::-1],[1,0],[.5,.5],r)
    assert a['ndcg10']==b['ndcg10']==1


def test_paired_bootstrap_identity():
    a=pd.DataFrame({'ndcg':[.2,.4,0],'n_relevant':[1,1,0]},index=[1,2,3])
    ci=paired_bootstrap(a,a,n=100)
    assert ci['delta']==ci['ci95_low']==ci['ci95_high']==0


def test_label_horizon_end_is_excluded(fixture):
    events,items,cut=fixture
    extra=pd.DataFrame({'user_id':[1],'item_id':[15],'timestamp':[cut+pd.Timedelta(days=7)]})
    h=history_before(events,cut); users=eligible_users(h,cut)
    cand,_=candidates(h,items,users,cut)
    y,r,_=attach_labels(events,h,cand,users,cut)
    yy,rr,_=attach_labels(pd.concat([events,extra]),h,cand,users,cut)
    np.testing.assert_array_equal(y,yy)
    pd.testing.assert_series_equal(r,rr)
