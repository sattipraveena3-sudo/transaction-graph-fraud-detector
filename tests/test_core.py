import pandas as pd
from app.core import evaluate,generate_transactions,graph_features,score_accounts
def test_graph_features_known_cycle():
    f=pd.DataFrame([('T1','A','B',10,1,'X',False),('T2','B','A',12,2,'X',True)],columns=['id','source','target','amount','timestamp','type','is_fraud']); g,x=graph_features(f); assert x.cycle_count.sum()>=2
def test_scoring_and_explanation():
    _,x=graph_features(generate_transactions(40,300)); s=score_accounts(x); assert s.score.between(0,2).all() and s.explanation.str.contains('cycles=').all()
def test_evaluation_is_real_and_bounded():
    _,x=graph_features(generate_transactions()); m=evaluate(score_accounts(x)); assert all(0<=v<=1 for v in m.values())
