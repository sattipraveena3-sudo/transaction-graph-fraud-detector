import networkx as nx, numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score,recall_score,roc_auc_score

def generate_transactions(accounts=120,transactions=900,seed=11):
    rng=np.random.default_rng(seed); rows=[]
    for i in range(transactions):
        src=f"A{rng.integers(accounts)}"; dst=f"A{rng.integers(accounts)}"; amount=float(rng.lognormal(4,1)); fraud=False
        if i<45:
            src=f"A{(i%5)}"; dst=f"A{((i+1)%5)}"; amount=float(rng.uniform(8000,15000)); fraud=True
        rows.append((f"T{i}",src,dst,amount,i,"TRANSFER",fraud))
    return pd.DataFrame(rows,columns=["id","source","target","amount","timestamp","type","is_fraud"])

def graph_features(frame):
    g=nx.DiGraph()
    for r in frame.itertuples():
        previous=g.get_edge_data(r.source,r.target,{}).get("amount",0); g.add_edge(r.source,r.target,amount=previous+r.amount)
    pagerank=nx.pagerank(g,weight="amount"); between=nx.betweenness_centrality(g); cycles={n:0 for n in g}
    for cycle in list(nx.simple_cycles(g,length_bound=5))[:2000]:
        for n in cycle: cycles[n]+=1
    grouped=frame.groupby("source").agg(total_amount=("amount","sum"),transaction_count=("id","count"),mean_amount=("amount","mean"),fraud_label=("is_fraud","max"))
    for name,values in [("pagerank",pagerank),("betweenness",between),("cycle_count",cycles)]: grouped[name]=grouped.index.map(values).fillna(0)
    return g,grouped.reset_index(names="account")

def score_accounts(features):
    columns=["total_amount","transaction_count","mean_amount","pagerank","betweenness","cycle_count"]
    model=IsolationForest(n_estimators=150,contamination=.08,random_state=11).fit(features[columns]); result=features.copy(); result["score"]=-model.score_samples(features[columns]); result["flagged"]=model.predict(features[columns])==-1
    result["explanation"]=result.apply(lambda r:f"amount={r.total_amount:.0f}; cycles={int(r.cycle_count)}; centrality={r.pagerank:.4f}; model_score={r.score:.4f}",axis=1)
    return result.sort_values("score",ascending=False)

def evaluate(scored):
    y=scored.fraud_label.astype(int); pred=scored.flagged.astype(int)
    return {"precision":float(precision_score(y,pred,zero_division=0)),"recall":float(recall_score(y,pred,zero_division=0)),"roc_auc":float(roc_auc_score(y,scored.score))}
