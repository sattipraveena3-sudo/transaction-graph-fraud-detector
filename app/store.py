import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import pandas as pd


class Store:
    def __init__(self, path: str = "data/fraud.db"):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.setup()

    def connect(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def setup(self):
        with self.connect() as conn:
            conn.execute("""
                create table if not exists transactions(
                    id text primary key,
                    source text not null,
                    target text not null,
                    amount real not null,
                    timestamp real not null,
                    type text not null,
                    is_fraud integer not null default 0
                )
            """)
            conn.execute("""
                create table if not exists investigations(
                    id integer primary key autoincrement,
                    account text not null,
                    status text not null,
                    note text not null default '',
                    created real not null
                )
            """)
            conn.execute("create index if not exists idx_tx_source on transactions(source)")
            conn.execute("create index if not exists idx_tx_target on transactions(target)")
            conn.execute("create index if not exists idx_tx_timestamp on transactions(timestamp desc)")

    def add_transaction(self, item: dict[str, Any]) -> dict[str, Any]:
        with self.connect() as conn:
            conn.execute(
                "insert or replace into transactions(id,source,target,amount,timestamp,type,is_fraud) values(?,?,?,?,?,?,?)",
                (
                    item["id"], item["source"], item["target"], float(item["amount"]),
                    float(item.get("timestamp") or time.time()), item.get("type", "TRANSFER"),
                    int(bool(item.get("is_fraud", False))),
                ),
            )
        return item

    def add_frame(self, frame: pd.DataFrame) -> int:
        count = 0
        for row in frame.to_dict("records"):
            self.add_transaction(row)
            count += 1
        return count

    def frame(self, limit: int = 5000) -> pd.DataFrame:
        with self.connect() as conn:
            rows = conn.execute(
                "select id,source,target,amount,timestamp,type,is_fraud from transactions order by timestamp desc limit ?",
                (max(1, min(limit, 20000)),),
            ).fetchall()
        if not rows:
            return pd.DataFrame(columns=["id","source","target","amount","timestamp","type","is_fraud"])
        frame = pd.DataFrame([dict(row) for row in rows])
        frame["is_fraud"] = frame["is_fraud"].astype(bool)
        return frame.sort_values("timestamp")

    def transaction_count(self) -> int:
        with self.connect() as conn:
            return int(conn.execute("select count(*) from transactions").fetchone()[0])

    def clear_transactions(self) -> None:
        with self.connect() as conn:
            conn.execute("delete from transactions")

    def add_investigation(self, account: str, status: str, note: str = "") -> dict[str, Any]:
        created = time.time()
        with self.connect() as conn:
            cur = conn.execute(
                "insert into investigations(account,status,note,created) values(?,?,?,?)",
                (account, status, note, created),
            )
            iid = int(cur.lastrowid)
        return {"id": iid, "account": account, "status": status, "note": note, "created": created}

    def investigations(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.connect() as conn:
            return [dict(row) for row in conn.execute(
                "select * from investigations order by created desc limit ?", (max(1, min(limit, 500)),)
            )]
