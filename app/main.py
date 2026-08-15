from pathlib import Path
from fastapi import FastAPI,HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from app.core import evaluate,generate_transactions,graph_features,score_accounts
transactions=generate_transactions(); graph,features=graph_features(transactions); scored=score_accounts(features); metrics=evaluate(scored)
app=FastAPI(title="Transaction Graph Fraud Detector"); static=Path(__file__).parent/"static"; app.mount("/static",StaticFiles(directory=static),name="static")
@app.get("/")
def home(): return FileResponse(static/"index.html")
@app.get("/health")
def health(): return {"status":"ok","evaluation":metrics}
@app.get("/alerts")
def alerts(limit:int=20): return scored.head(limit).to_dict("records")
@app.get("/account/{account}/explanation")
def explanation(account:str):
    row=scored[scored.account==account]
    if row.empty: raise HTTPException(404,"account not found")
    return row.iloc[0].to_dict()
@app.get("/graph")
def graph_data(): return {"nodes":[{"id":n,"flagged":bool(scored.set_index('account').flagged.get(n,False))} for n in graph.nodes],"edges":[{"source":a,"target":b} for a,b in list(graph.edges)[:300]]}
