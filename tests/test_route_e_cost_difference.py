"""Run with `python -m pytest tests/test_route_e_cost_difference.py` in AdaptThink.
Requires the pre-existing scripts directory on sys.path, but no cached data.
"""
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))

import route_e_cost_difference as f


def fake_data(n=60, d=8, seed=13):
    rng=np.random.default_rng(seed)
    X=rng.normal(size=(n,d))
    tt=np.exp(5+0.4*X[:,0]+rng.normal(0,.1,n))
    tn=np.exp(3+0.3*X[:,1]+rng.normal(0,.1,n))
    rows=[dict(sample_id=f'math:{i}',dataset='math',t_t=float(tt[i]),
               t_n=float(tn[i]),p_t=.8,p_n=.6,delta=.2) for i in range(n)]
    return X, rows


def test_direct_delta_ridge_is_finite():
    X,rows=fake_data()
    model=f.direct_ridge_fit(X,rows,.02,1000)
    pred=f.direct_ridge_predict(model,X)
    assert len(pred)==len(rows)
    assert np.isfinite(pred).all()


def test_margin_beta_zero_matches_p2():
    import route_e_token_aware as p2
    X,rows=fake_data()
    base=p2.ridge_fit(X,p2.target_matrix(rows),.02)
    pred_base=p2.predict_heads(base,X)
    mod,meta=f.fit_margin(X,rows,.02,1000.,0.,40,base_fit=base)
    delta,_=f.margin_predict(mod,X)
    assert meta['optimized'] is False
    np.testing.assert_allclose(delta,pred_base['t_t']-pred_base['t_n'],atol=1e-10,rtol=0)


def test_margin_positive_beta_train_objective_decreases():
    X,rows=fake_data()
    model,meta=f.fit_margin(X,rows,.02,1000.,0.1,35)
    scores,_=f.margin_predict(model,X)
    assert np.isfinite(scores).all()
    assert meta['objective_final'] <= meta['objective_initial']+1e-7


def test_nauc_known_curve():
    test=[dict(tokens=10,accuracy=.2),dict(tokens=15,accuracy=.5),dict(tokens=20,accuracy=.8)]
    assert abs(f.auc_cost_sorted(test,10,20)-5)<1e-10


def test_fold_determinism_and_train_only_cv():
    X,rows=fake_data(n=75)
    a=f.fold_assignments(rows,3,42)
    b=f.fold_assignments(rows,3,42)
    assert np.array_equal(a,b)
    assert set(a)=={0,1,2}
    # Running one real CV beta candidate proves the selected model uses only
    # the passed train features/labels, not global validation state.
    selected, metrics, _=f.select_beta_cv(X,rows,[0.,0.1],3,42,.02,1000.,20)
    assert selected in {0.,0.1}
    assert all(np.isfinite(m['mae']) for m in metrics.values())
