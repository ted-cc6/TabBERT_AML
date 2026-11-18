"""Prepare causal account sequences and baseline features for IBM AML HI-Small.

Run the stages in order. Keep the temporary SQLite database outside the repository checkout.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import random
import sqlite3
from collections import Counter, deque
from datetime import date
from pathlib import Path

import numpy as np

RAW_COLUMNS = [
    "Timestamp", "From Bank", "Account", "To Bank", "Account",
    "Amount Received", "Receiving Currency", "Amount Paid",
    "Payment Currency", "Payment Format", "Is Laundering",
]
TOKEN_FIELDS = [
    "payment_format", "currency", "counterparty_bank", "direction",
    "amount_bucket", "time_gap_bucket",
]
DATASET_URL = "https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml"
REFERENCE_URL = "https://github.com/IBM/Multi-GNN/blob/main/format_kaggle_files.py"
SEEDS = (42, 43, 44)


def log(message: str) -> None:
    print(message, flush=True)


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=OFF")
    con.execute("PRAGMA synchronous=OFF")
    con.execute("PRAGMA temp_store=FILE")
    con.execute("PRAGMA cache_size=-200000")
    return con


def read_patterns(path: Path) -> dict[tuple[str, ...], set[str]]:
    patterns: dict[tuple[str, ...], set[str]] = {}
    current = None
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            line = line.strip()
            if line.startswith("BEGIN LAUNDERING ATTEMPT - "):
                current = line.split(" - ", 1)[1].split(":", 1)[0].strip().upper()
            elif line.startswith("END LAUNDERING ATTEMPT"):
                current = None
            elif line and current:
                row = next(csv.reader([line]))
                if len(row) == 11:
                    patterns.setdefault(tuple(row), set()).add(current)
    return patterns


def minute_key(value: str) -> int:
    # Preserve the source clock without guessing a time zone.
    day = date(int(value[:4]), int(value[5:7]), int(value[8:10])).toordinal()
    return day * 1440 + int(value[11:13]) * 60 + int(value[14:16])


def format_minute(value: int) -> str:
    day, minute = divmod(value, 1440)
    return f"{date.fromordinal(day).isoformat()} {minute // 60:02d}:{minute % 60:02d}"


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_state(root: Path) -> dict:
    return json.loads((root / "processed" / "pipeline_state.json").read_text(encoding="utf-8"))


def split_name(timestamp_minute: int, state: dict) -> str:
    if timestamp_minute >= state["primary_end_exclusive_minute"]:
        return "tail"
    if timestamp_minute < state["train_end_exclusive_minute"]:
        return "train"
    if timestamp_minute < state["valid_end_exclusive_minute"]:
        return "valid"
    return "test"


def ingest(root: Path, con: sqlite3.Connection) -> None:
    raw = root / "raw" / "HI-Small_Trans.csv.gz"
    if not raw.exists():
        raw = root / "raw" / "HI-Small_Trans.csv"
    pattern_path = root / "raw" / "HI-Small_Patterns.txt"
    if not raw.exists() or not pattern_path.exists():
        raise FileNotFoundError("Missing HI-Small transactions or patterns in raw/")
    patterns = read_patterns(pattern_path)
    con.executescript("""
        DROP TABLE IF EXISTS events;
        DROP TABLE IF EXISTS transactions;
        CREATE TABLE transactions (
            transaction_id INTEGER PRIMARY KEY, timestamp_text TEXT NOT NULL,
            timestamp_minute INTEGER NOT NULL, date_text TEXT NOT NULL,
            from_bank TEXT NOT NULL, from_account TEXT NOT NULL,
            to_bank TEXT NOT NULL, to_account TEXT NOT NULL,
            amount_received REAL NOT NULL, receiving_currency TEXT NOT NULL,
            amount_paid REAL NOT NULL, payment_currency TEXT NOT NULL,
            payment_format TEXT NOT NULL, is_laundering INTEGER NOT NULL,
            typology TEXT
        );
    """)
    opener = gzip.open if raw.suffix == ".gz" else open
    missing = Counter()
    matched_keys = set()
    invalid = []
    batch = []
    with opener(raw, "rt", newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        if header != RAW_COLUMNS:
            raise ValueError(f"Unexpected raw header: {header}")
        for transaction_id, row in enumerate(reader):
            if len(row) != 11:
                invalid.append((transaction_id, "column_count"))
                continue
            for name, value in zip(RAW_COLUMNS, row):
                if value == "":
                    missing[name] += 1
            if any(value == "" for value in row):
                invalid.append((transaction_id, "missing_value"))
                continue
            try:
                received = float(row[5])
                paid = float(row[7])
                label = int(row[10])
                timestamp_minute = minute_key(row[0])
                if not math.isfinite(received) or not math.isfinite(paid):
                    raise ValueError("nonfinite_amount")
                if received < 0 or paid < 0 or label not in (0, 1):
                    raise ValueError("invalid_amount_or_label")
            except (ValueError, IndexError) as exc:
                invalid.append((transaction_id, str(exc)))
                continue
            key = tuple(row)
            typology = "|".join(sorted(patterns[key])) if key in patterns else None
            if typology:
                matched_keys.add(key)
            batch.append((
                transaction_id, row[0], timestamp_minute, row[0][:10],
                row[1], row[2], row[3], row[4], received, row[6],
                paid, row[8], row[9], label, typology,
            ))
            if len(batch) >= 50000:
                con.executemany("INSERT INTO transactions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", batch)
                con.commit()
                batch.clear()
                if (transaction_id + 1) % 500000 < 50000:
                    log(f"Read {transaction_id + 1:,} transactions")
    if batch:
        con.executemany("INSERT INTO transactions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", batch)
        con.commit()
    if invalid:
        raise ValueError(f"Found {len(invalid)} invalid rows; first five: {invalid[:5]}")
    unmatched = len(patterns.keys() - matched_keys)
    if unmatched:
        raise ValueError(f"Failed to match {unmatched} pattern rows to source transactions")
    con.executescript("""
        CREATE INDEX tx_time_idx ON transactions(timestamp_minute, transaction_id);
        CREATE TABLE events (
            event_id INTEGER PRIMARY KEY, transaction_id INTEGER NOT NULL,
            account_key TEXT NOT NULL, timestamp_minute INTEGER NOT NULL,
            direction INTEGER NOT NULL, amount REAL NOT NULL,
            currency TEXT NOT NULL, counterparty_bank TEXT NOT NULL,
            counterparty_key TEXT NOT NULL, payment_format TEXT NOT NULL
        );
        INSERT INTO events
        SELECT transaction_id * 2, transaction_id,
               from_bank || ':' || from_account, timestamp_minute, 0,
               amount_paid, payment_currency, to_bank,
               to_bank || ':' || to_account, payment_format
        FROM transactions;
        INSERT INTO events
        SELECT transaction_id * 2 + 1, transaction_id,
               to_bank || ':' || to_account, timestamp_minute, 1,
               amount_received, receiving_currency, from_bank,
               from_bank || ':' || from_account, payment_format
        FROM transactions;
        CREATE INDEX events_account_time_idx
            ON events(account_key, timestamp_minute, transaction_id, direction);
    """)
    con.commit()
    n, positives, minimum, maximum = con.execute(
        "SELECT COUNT(*), SUM(is_laundering), MIN(timestamp_minute), MAX(timestamp_minute) FROM transactions"
    ).fetchone()
    primary_end = date(2022, 9, 11).toordinal() * 1440
    span = primary_end - minimum
    state = {
        "dataset": "IBM AML HI-Small",
        "dataset_url": DATASET_URL,
        "reference_script": REFERENCE_URL,
        "raw_sha256": {
            raw.name: sha256_file(raw),
            pattern_path.name: sha256_file(pattern_path),
        },
        "transaction_count": n,
        "positive_count": positives,
        "event_count": n * 2,
        "matched_pattern_row_keys": len(matched_keys),
        "missing_values": dict(missing),
        "min_timestamp_minute": minimum,
        "max_timestamp_minute": maximum,
        "primary_end_exclusive_minute": primary_end,
        "train_end_exclusive_minute": minimum + int(span * 0.60),
        "valid_end_exclusive_minute": minimum + int(span * 0.80),
    }
    write_json(root / "processed" / "pipeline_state.json", state)
    log(f"Ingested {n:,} transactions; {positives:,} positive; {n*2:,} account events")


def percentile(sorted_values: list[int], fraction: float) -> int:
    return sorted_values[min(len(sorted_values) - 1, int((len(sorted_values) - 1) * fraction))]


def eda(root: Path, con: sqlite3.Connection) -> None:
    state = load_state(root)
    primary_end = date(2022, 9, 11).toordinal() * 1440
    primary_span = primary_end - state["min_timestamp_minute"]
    state["primary_end_exclusive_minute"] = primary_end
    state["train_end_exclusive_minute"] = state["min_timestamp_minute"] + int(primary_span * 0.60)
    state["valid_end_exclusive_minute"] = state["min_timestamp_minute"] + int(primary_span * 0.80)
    daily = [
        dict(zip(("date", "transactions", "laundering"), row))
        for row in con.execute(
            "SELECT date_text, COUNT(*), SUM(is_laundering) FROM transactions GROUP BY date_text ORDER BY date_text"
        )
    ]
    split_rows = [
        dict(zip(("split", "transactions", "laundering"), row))
        for row in con.execute("""
            SELECT CASE WHEN timestamp_minute >= ? THEN 'tail'
                        WHEN timestamp_minute < ? THEN 'train'
                        WHEN timestamp_minute < ? THEN 'valid' ELSE 'test' END,
                   COUNT(*), SUM(is_laundering)
            FROM transactions GROUP BY 1 ORDER BY 1
        """, (primary_end, state["train_end_exclusive_minute"], state["valid_end_exclusive_minute"]))
    ]
    typologies = [
        dict(zip(("typology", "transactions"), row))
        for row in con.execute("""
            SELECT COALESCE(typology, 'UNTAGGED'), COUNT(*)
            FROM transactions WHERE is_laundering=1 GROUP BY 1 ORDER BY 2 DESC
        """)
    ]
    currencies = [
        dict(zip(("currency", "transactions"), row))
        for row in con.execute(
            "SELECT payment_currency, COUNT(*) FROM transactions GROUP BY 1 ORDER BY 2 DESC"
        )
    ]
    formats = [
        dict(zip(("payment_format", "transactions"), row))
        for row in con.execute(
            "SELECT payment_format, COUNT(*) FROM transactions GROUP BY 1 ORDER BY 2 DESC"
        )
    ]
    lengths = []
    train_lengths = []
    account = None
    account_length = 0
    train_length = 0
    previous_time = None
    max_train_gap = 0
    train_end = state["train_end_exclusive_minute"]
    for current_account, timestamp_minute in con.execute("""
        SELECT account_key, timestamp_minute FROM events
        ORDER BY account_key, timestamp_minute, transaction_id, direction
    """):
        if current_account != account:
            if account is not None:
                lengths.append(account_length)
                if train_length:
                    train_lengths.append(train_length)
            account = current_account
            account_length = 0
            train_length = 0
            previous_time = None
        if previous_time is not None and timestamp_minute < train_end:
            max_train_gap = max(max_train_gap, timestamp_minute - previous_time)
        previous_time = timestamp_minute
        account_length += 1
        if timestamp_minute < train_end:
            train_length += 1
    if account is not None:
        lengths.append(account_length)
        if train_length:
            train_lengths.append(train_length)
    lengths.sort()
    train_lengths.sort()
    quantiles = {
        "min": lengths[0], "p50": percentile(lengths, 0.50),
        "p90": percentile(lengths, 0.90), "p95": percentile(lengths, 0.95),
        "p99": percentile(lengths, 0.99), "max": lengths[-1],
    }
    train_quantiles = {
        "min": train_lengths[0], "p50": percentile(train_lengths, 0.50),
        "p90": percentile(train_lengths, 0.90), "p95": percentile(train_lengths, 0.95),
        "p99": percentile(train_lengths, 0.99), "max": train_lengths[-1],
    }
    sequence_length = 16 if train_quantiles["p90"] <= 16 else 32 if train_quantiles["p90"] <= 32 else 64
    max_train_amount = con.execute("""
        SELECT MAX(MAX(amount_paid, amount_received)) FROM transactions
        WHERE timestamp_minute < ?
    """, (train_end,)).fetchone()[0]
    amount_sample = sorted(
        value for (value,) in con.execute(
            "SELECT amount_paid FROM transactions WHERE transaction_id % 50 = 0"
        )
    )
    amount_min, amount_mean, amount_max = con.execute(
        "SELECT MIN(amount_paid), AVG(amount_paid), MAX(amount_paid) FROM transactions"
    ).fetchone()
    tail_count, tail_positive = con.execute("""
        SELECT COUNT(*), COALESCE(SUM(is_laundering),0) FROM transactions
        WHERE timestamp_minute >= ?
    """, (primary_end,)).fetchone()
    summary = {
        "transaction_count": state["transaction_count"],
        "positive_count": state["positive_count"],
        "positive_rate": state["positive_count"] / state["transaction_count"],
        "account_count": len(lengths),
        "date_min": daily[0]["date"],
        "date_max": daily[-1]["date"],
        "daily": daily,
        "split_counts": split_rows,
        "typology_counts": typologies,
        "payment_currency_counts": currencies,
        "payment_format_counts": formats,
        "missing_values": state["missing_values"],
        "amount_paid": {
            "min": amount_min, "mean": amount_mean, "max": amount_max,
            "sample_every_50th_row": len(amount_sample),
            "sample_p50": percentile(amount_sample, 0.50),
            "sample_p95": percentile(amount_sample, 0.95),
            "sample_p99": percentile(amount_sample, 0.99),
        },
        "account_event_length": quantiles,
        "train_account_event_length": train_quantiles,
        "recommended_sequence_length": sequence_length,
        "max_train_gap_minutes": max_train_gap,
        "max_train_amount": max_train_amount,
        "after_2022_09_10": {"transactions": tail_count, "laundering": tail_positive},
    }
    write_json(root / "reports" / "eda_summary.json", summary)
    state.update({
        "account_count": len(lengths),
        "sequence_length": sequence_length,
        "max_train_gap_minutes": max_train_gap,
        "max_train_amount": max_train_amount,
    })
    write_json(root / "processed" / "pipeline_state.json", state)
    write_eda_report(root, summary)
    write_charts(root, summary, lengths)
    log(f"EDA: {len(lengths):,} accounts; train p90={train_quantiles['p90']}; window={sequence_length}; tail={tail_count:,}")


def write_eda_report(root: Path, summary: dict) -> None:
    split_table = "\n".join(
        f"| {r['split']} | {r['transactions']:,} | {r['laundering']:,} | {r['laundering']/r['transactions']:.4%} |"
        for r in summary["split_counts"]
    )
    typology_table = "\n".join(
        f"| {r['typology']} | {r['transactions']:,} |" for r in summary["typology_counts"]
    )
    format_table = "\n".join(
        f"| {r['payment_format']} | {r['transactions']:,} |"
        for r in summary["payment_format_counts"]
    )
    tail = summary["after_2022_09_10"]
    amount = summary["amount_paid"]
    text = f"""# HI-Small EDA

Render this file in GitHub or a Markdown preview to display the figures from `figures/`.

## Check size and time coverage

- Read {summary['transaction_count']:,} transactions across {summary['account_count']:,} accounts; flag {summary['positive_count']:,} laundering transactions ({summary['positive_rate']:.4%}).
- Cover {summary['date_min']} through {summary['date_max']}; the source timestamps do not specify a time zone.
- Check source columns: {', '.join(RAW_COLUMNS)}. The two Account columns identify the sender and receiver, respectively.
- Check missing values: {summary['missing_values'] or 'zero in every column'}.
- Inspect the tail after 2022-09-10: {tail['transactions']:,} transactions, including {tail['laundering']:,} laundering transactions.
- Split the main September 1–10 period by elapsed time into 60% train, 20% validation, and 20% test. Keep September 11 onward as a separate tail audit set.

| split | transactions | laundering | rate |
| --- | ---: | ---: | ---: |
{split_table}

### Plot daily activity

![Daily transaction counts](../figures/daily_transactions.svg)

![Daily laundering transaction counts](../figures/daily_laundering.svg)

![Daily laundering rate](../figures/daily_laundering_rate.svg)

## Check account sequences

- Count sender and receiver events; each transaction yields two account events.
- Record event-count quantiles across all accounts: {summary['account_event_length']}.
- Record training-period account event-count quantiles: {summary['train_account_event_length']}.
- Select a fixed window length of {summary['recommended_sequence_length']} from 16, 32, and 64 using the training-period 90th percentile.

![Account-event sequence lengths and recommended window](../figures/account_sequence_lengths.svg)

## Check amounts and payment formats

- Record paid-amount minimum {amount['min']:.6g}, mean {amount['mean']:.6g}, and maximum {amount['max']:.6g}.
- Sample every 50th row for a deterministic scale check ({amount['sample_every_50th_row']:,} rows): p50={amount['sample_p50']:.6g}, p95={amount['sample_p95']:.6g}, p99={amount['sample_p99']:.6g}.
- Aggregate historical amounts only within the same currency.

| payment format | transactions |
| --- | ---: |
{format_table}

## Check laundering typologies

| typology | labelled transactions |
| --- | ---: |
{typology_table}

- Interpret UNTAGGED as laundering transactions absent from the patterns file, not as non-laundering transactions.

![Laundering typology counts](../figures/typology_counts.svg)

## Cite sources

- Dataset: {DATASET_URL}
- Preprocessing reference: {REFERENCE_URL}
"""
    (root / "reports" / "EDA_REPORT.md").write_text(text, encoding="utf-8")


def svg_bar_chart(title: str, labels: list[str], values: list[float], path: Path, color: str, percent: bool = False) -> None:
    from html import escape

    width, left, top, row_height = 900, 240, 70, 36
    height = top + len(labels) * row_height + 35
    scale = (width - left - 135) / max(max(values or [0]), 1e-12)
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="20" y="35" font-family="Arial" font-size="22" font-weight="bold">{escape(title)}</text>',
    ]
    for i, (label, value) in enumerate(zip(labels, values)):
        y = top + i * row_height
        body.append(f'<text x="20" y="{y+17}" font-family="Arial" font-size="14">{escape(label)}</text>')
        body.append(f'<rect x="{left}" y="{y}" width="{max(1, value*scale):.1f}" height="22" fill="{color}"/>')
        value_text = f"{value:.3f}%" if percent else f"{value:,}"
        body.append(f'<text x="{left+value*scale+8:.1f}" y="{y+17}" font-family="Arial" font-size="13">{value_text}</text>')
    body.append("</svg>")
    path.write_text("\n".join(body), encoding="utf-8")


def write_charts(root: Path, summary: dict, lengths: list[int]) -> None:
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    svg_bar_chart("Daily transactions", [x["date"] for x in summary["daily"]],
                  [x["transactions"] for x in summary["daily"]],
                  figures / "daily_transactions.svg", "#2c6f9f")
    svg_bar_chart("Daily laundering transactions", [x["date"] for x in summary["daily"]],
                  [x["laundering"] for x in summary["daily"]],
                  figures / "daily_laundering.svg", "#b14a4a")
    svg_bar_chart("Daily laundering rate", [x["date"] for x in summary["daily"]],
                  [100 * x["laundering"] / x["transactions"] for x in summary["daily"]],
                  figures / "daily_laundering_rate.svg", "#a35377", percent=True)
    bins = [
        ("1", 1, 1), ("2-4", 2, 4), ("5-8", 5, 8), ("9-16", 9, 16),
        ("17-32", 17, 32), ("33-64", 33, 64), ("65+", 65, 10**12),
    ]
    counts = [sum(lo <= length <= hi for length in lengths) for _, lo, hi in bins]
    svg_bar_chart(f"Account event sequence lengths (recommended window: {summary['recommended_sequence_length']})", [x[0] for x in bins],
                  counts, figures / "account_sequence_lengths.svg", "#4e8070")
    svg_bar_chart("Laundering typologies", [x["typology"] for x in summary["typology_counts"]],
                  [x["transactions"] for x in summary["typology_counts"]],
                  figures / "typology_counts.svg", "#8b6596")


def make_vocab(con: sqlite3.Connection, column: str, train_end: int) -> dict[str, int]:
    if column not in {"payment_format", "currency", "counterparty_bank"}:
        raise ValueError(column)
    rows = con.execute(f"""
        SELECT {column}, COUNT(*) FROM events WHERE timestamp_minute < ?
        GROUP BY {column} ORDER BY COUNT(*) DESC, {column}
    """, (train_end,))
    values = {value: index for index, (value, _) in enumerate(rows, start=3)}
    return {"<PAD>": 0, "<UNK>": 1, "<MASK>": 2, **values}


def bucket(value: float | None, max_value: float, bins: int) -> int:
    if value is None:
        return 3
    if value < 0 or not math.isfinite(value):
        return 1
    if max_value <= 0:
        return 4
    return 4 + min(bins - 1, int(math.log1p(value) / math.log1p(max_value) * bins))


class RecentWindow:
    def __init__(self, width_minutes: int):
        self.width = width_minutes
        self.items = deque()
        self.counterparties = Counter()
        self.formats = Counter()
        self.currency_count = Counter()
        self.currency_sum = Counter()
        self.sent = 0
        self.received = 0

    def expire(self, now: int) -> None:
        while self.items and now - self.items[0][0] > self.width:
            timestamp, direction, amount, currency, counterparty, payment_format = self.items.popleft()
            self.counterparties[counterparty] -= 1
            if not self.counterparties[counterparty]:
                del self.counterparties[counterparty]
            self.formats[payment_format] -= 1
            if not self.formats[payment_format]:
                del self.formats[payment_format]
            self.currency_count[currency] -= 1
            self.currency_sum[currency] -= amount
            if direction == 0:
                self.sent -= 1
            else:
                self.received -= 1

    def add(self, item: tuple) -> None:
        timestamp, direction, amount, currency, counterparty, payment_format = item
        self.items.append(item)
        self.counterparties[counterparty] += 1
        self.formats[payment_format] += 1
        self.currency_count[currency] += 1
        self.currency_sum[currency] += amount
        if direction == 0:
            self.sent += 1
        else:
            self.received += 1


FEATURE_NAMES = [
    "transaction_id", "split", "current_amount_paid", "current_log_amount_paid",
    "current_payment_currency", "current_payment_format", "same_bank", "self_transfer",
    "previous_event_gap_min", "prior_events_total", "events_1h", "events_24h", "events_7d",
    "sent_24h", "received_24h", "distinct_counterparties_24h",
    "distinct_counterparties_7d", "same_currency_amount_count_24h",
    "same_currency_amount_sum_24h", "same_currency_amount_mean_24h",
    "amount_to_prior_mean_ratio_24h", "current_format_share_24h",
    "mean_interevent_gap_24h",
]


def export(root: Path, con: sqlite3.Connection) -> None:
    state = load_state(root)
    if "sequence_length" not in state:
        raise RuntimeError("Run EDA before export")
    train_end = state["train_end_exclusive_minute"]
    vocabs = {
        column: make_vocab(con, column, train_end)
        for column in ("payment_format", "currency", "counterparty_bank")
    }
    vocabs["direction"] = {"<PAD>": 0, "<UNK>": 1, "<MASK>": 2, "sent": 3, "received": 4}
    write_json(root / "processed" / "vocab_and_bins.json", {
        "token_field_order": TOKEN_FIELDS,
        "special_ids": {"PAD": 0, "UNK": 1, "MASK": 2, "NO_HISTORY": 3},
        "categorical": vocabs,
        "amount_bucket": {"bins": 64, "train_max": state["max_train_amount"], "first_bin_id": 4},
        "time_gap_bucket": {"bins": 32, "train_max_minutes": state["max_train_gap_minutes"], "first_bin_id": 4},
        "fit_period": "train only",
    })
    processed = root / "processed"
    tokens = np.lib.format.open_memmap(
        processed / "event_tokens.npy", mode="w+", dtype=np.int32,
        shape=(state["event_count"], len(TOKEN_FIELDS)),
    )
    meta_file = gzip.open(processed / "events_meta.csv.gz", "wt", newline="", encoding="utf-8")
    windows_file = gzip.open(processed / "window_index.csv.gz", "wt", newline="", encoding="utf-8")
    features_file = gzip.open(processed / "engineered_features.csv.gz", "wt", newline="", encoding="utf-8")
    offsets_file = gzip.open(processed / "account_offsets.csv.gz", "wt", newline="", encoding="utf-8")
    meta = csv.writer(meta_file)
    windows = csv.writer(windows_file)
    features = csv.writer(features_file)
    offsets = csv.writer(offsets_file)
    meta.writerow(["event_index", "transaction_id", "account_key", "timestamp_minute", "direction", "split"])
    windows.writerow(["transaction_id", "target_event_index", "window_start_index",
                      "window_length", "split", "is_laundering", "typology"])
    features.writerow(FEATURE_NAMES)
    offsets.writerow(["account_key", "start_event_index", "event_count"])
    query = """
        SELECT e.transaction_id, e.account_key, e.timestamp_minute, e.direction,
               e.amount, e.currency, e.counterparty_bank, e.counterparty_key,
               e.payment_format, t.is_laundering, t.typology,
               t.from_bank, t.to_bank, t.from_account, t.to_account,
               t.amount_paid, t.payment_currency, t.payment_format
        FROM events e JOIN transactions t USING(transaction_id)
        ORDER BY e.account_key, e.timestamp_minute, e.transaction_id, e.direction
    """
    account = None
    account_start = 0
    account_count = 0
    prior_total = 0
    prior_time = None
    w1 = w24 = w7 = None
    event_index = 0
    targets = 0
    for row in con.execute(query):
        (transaction_id, account_value, timestamp_minute, direction, amount, currency,
         counterparty_bank, counterparty_key, payment_format, label, typology,
         from_bank, to_bank, from_account, to_account, amount_paid,
         payment_currency, current_format) = row
        if account_value != account:
            if account is not None:
                offsets.writerow([account, account_start, account_count])
            account = account_value
            account_start = event_index
            account_count = 0
            prior_total = 0
            prior_time = None
            w1, w24, w7 = RecentWindow(60), RecentWindow(1440), RecentWindow(10080)
        for window in (w1, w24, w7):
            window.expire(timestamp_minute)
        gap = None if prior_time is None else timestamp_minute - prior_time
        tokens[event_index] = [
            vocabs["payment_format"].get(payment_format, 1),
            vocabs["currency"].get(currency, 1),
            vocabs["counterparty_bank"].get(counterparty_bank, 1),
            3 if direction == 0 else 4,
            bucket(amount, state["max_train_amount"], 64),
            bucket(gap, state["max_train_gap_minutes"], 32),
        ]
        split = split_name(timestamp_minute, state)
        meta.writerow([event_index, transaction_id, account_value, timestamp_minute, direction, split])
        if direction == 0:
            length = min(state["sequence_length"], event_index - account_start + 1)
            windows.writerow([
                transaction_id, event_index, event_index - length + 1,
                length, split, label, typology or "",
            ])
            count_currency = w24.currency_count[payment_currency]
            sum_currency = w24.currency_sum[payment_currency]
            mean_currency = sum_currency / count_currency if count_currency else 0.0
            mean_gap = (
                (w24.items[-1][0] - w24.items[0][0]) / (len(w24.items) - 1)
                if len(w24.items) > 1 else -1
            )
            features.writerow([
                transaction_id, split, amount_paid, math.log1p(amount_paid),
                payment_currency, current_format, int(from_bank == to_bank),
                int(from_bank == to_bank and from_account == to_account),
                gap if gap is not None else -1, prior_total,
                len(w1.items), len(w24.items), len(w7.items),
                w24.sent, w24.received, len(w24.counterparties),
                len(w7.counterparties), count_currency, sum_currency,
                mean_currency, amount_paid / mean_currency if mean_currency > 0 else 0.0,
                w24.formats[current_format] / len(w24.items) if w24.items else 0.0,
                mean_gap,
            ])
            targets += 1
        item = (timestamp_minute, direction, amount, currency, counterparty_key, payment_format)
        for window in (w1, w24, w7):
            window.add(item)
        prior_total += 1
        prior_time = timestamp_minute
        event_index += 1
        account_count += 1
        if event_index % 1000000 == 0:
            log(f"Exported {event_index:,} account events")
    if account is not None:
        offsets.writerow([account, account_start, account_count])
    for handle in (meta_file, windows_file, features_file, offsets_file):
        handle.close()
    tokens.flush()
    del tokens
    if event_index != state["event_count"] or targets != state["transaction_count"]:
        raise AssertionError(f"Export mismatch: {event_index} events, {targets} targets")
    np.savez_compressed(
        processed / "event_tokens.npz",
        tokens=np.load(processed / "event_tokens.npy", mmap_mode="r"),
    )
    export_clean_transactions(root, con, state)
    write_contract(root, state)
    log(f"Exported {event_index:,} events and {targets:,} targets")


def export_clean_transactions(root: Path, con: sqlite3.Connection, state: dict) -> None:
    path = root / "processed" / "transactions_clean.csv.gz"
    with gzip.open(path, "wt", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "transaction_id", "timestamp_text", "timestamp_minute",
            "from_account_key", "to_account_key", "amount_received",
            "receiving_currency", "amount_paid", "payment_currency",
            "payment_format", "split", "is_laundering", "typology",
        ])
        for row in con.execute("""
            SELECT transaction_id, timestamp_text, timestamp_minute,
                   from_bank || ':' || from_account, to_bank || ':' || to_account,
                   amount_received, receiving_currency, amount_paid,
                   payment_currency, payment_format, is_laundering, typology
            FROM transactions ORDER BY transaction_id
        """):
            writer.writerow([*row[:10], split_name(row[2], state), row[10], row[11] or ""])


def write_contract(root: Path, state: dict) -> None:
    text = f"""# Portion 1 data contract

This document explains how to use the processed datasets, including file formats, field definitions, join keys, split dates, window loading, and leakage guards.

For an overview and visualizations of the source data, see the [EDA report](reports/EDA_REPORT.md).

## Fixed data and splits

- Read raw/HI-Small_Trans.csv.gz and raw/HI-Small_Patterns.txt; retain both source files.
- Set transaction_id to the zero-based data-row number in the source CSV.
- Identify each account as bank:account; create one sender event and one receiver event per transaction.
- Keep the source timestamps at minute precision; do not infer a time zone.
- Assign train to [{format_minute(state['min_timestamp_minute'])}, {format_minute(state['train_end_exclusive_minute'])}), valid to [{format_minute(state['train_end_exclusive_minute'])}, {format_minute(state['valid_end_exclusive_minute'])}), test to [{format_minute(state['valid_end_exclusive_minute'])}, {format_minute(state['primary_end_exclusive_minute'])}), and all remaining rows to tail.
- Count {state['transaction_count']:,} transactions, {state['event_count']:,} events, and {state['account_count']:,} accounts.
- Fix the window at {state['sequence_length']} events. Include the current target event and earlier account events only.
- Use each transaction's sender event as its single prediction target; retain the receiver event in account history.
- Break same-minute ties by transaction_id, then sent(0) / received(1). This is a stable ordering convention, not evidence of actual transaction order.

## Handoff files

| File | Content | Use |
| --- | --- | --- |
| [`processed/transactions_clean.csv.gz`](processed/transactions_clean.csv.gz) | One row per transaction | Inspect normalized account keys, split, label, and typology |
| [`processed/event_tokens.npz`](processed/event_tokens.npz) | {state['event_count']:,} × 6 int32 values | Read field tokens by event_index; stored in compressed form |
| [`processed/events_meta.csv.gz`](processed/events_meta.csv.gz) | One row per event | Inspect event position, account, time, and direction |
| [`processed/window_index.csv.gz`](processed/window_index.csv.gz) | One row per transaction | Read the target event and causal window start |
| [`processed/engineered_features.csv.gz`](processed/engineered_features.csv.gz) | One row per transaction | Train LightGBM / Isolation Forest; excludes labels |
| [`processed/account_offsets.csv.gz`](processed/account_offsets.csv.gz) | One row per account | Locate each account's event range |
| [`processed/vocab_and_bins.json`](processed/vocab_and_bins.json) | JSON | Inspect training-period vocabularies, numeric bins, and special tokens |
| [`processed/label_subsets.csv.gz`](processed/label_subsets.csv.gz) | Subset membership rows | Find train IDs for 1%, 5%, and 20% across seeds; add 2% when needed |
| [`processed/label_subset_100pct.csv.gz`](processed/label_subset_100pct.csv.gz) | One row per train transaction | Reuse the full train ID and label list for every seed |
| [`processed/label_subset_manifest.json`](processed/label_subset_manifest.json) | JSON | Inspect subset counts and the full-train list path |

## Read a window

Load np.load("processed/event_tokens.npz")["tokens"]. Slice
tokens[window_start_index:target_event_index+1] and left-pad with PAD=0 to {state['sequence_length']} events.
For lower memory use, extract event_tokens.npy from the archive and open it with mmap_mode="r".

## Prevent leakage

- Build category vocabularies and amount/time-gap bins from train events only.
- Build windows from the target and earlier events only; compute baseline history features from earlier account events only.
- Keep is_laundering and typology in target files; exclude both from tokens and baseline features.
- Sample label subsets within train, stratify by class, and nest 1% within 5% and 5% within 20% for each seed.
- Add a 2% subset if the 1% subset contains fewer than 30 positive examples; record this in the manifest.
- Use the same 100% train list for every seed; do not duplicate identical rows.
- Choose decision thresholds on valid before computing test metrics.

## Field definitions

- event_tokens.npz tokens column order: {', '.join(TOKEN_FIELDS)}.
- Join windows, labels, and features by transaction_id. Their row orders are not guaranteed to match.
- engineered_features.csv.gz column order: {', '.join(FEATURE_NAMES)}.
- Use 1-hour, 24-hour, and 7-day windows over events before the current target event.
- Aggregate monetary amounts only within the same currency; cross-currency history contributes counts, directions, and counterparties only.
- Match typology exactly against pattern-file rows. An empty typology means absent from that file, not a negative label.

## References

- Dataset: {DATASET_URL}
- Adapt the bank/account node-key and temporal-ordering ideas from IBM Multi-GNN's format_kaggle_files.py. Build a separate sequence pipeline; do not run the graph-model formatter directly.
- Keep the post-2022-09-10 tail separate from the main test metrics, consistent with the Kaggle dataset note.
"""
    (root / "DATA_CONTRACT.md").write_text(text, encoding="utf-8")


def subsets(root: Path, con: sqlite3.Connection) -> None:
    state = load_state(root)
    positives, negatives = [], []
    full_path = root / "processed" / "label_subset_100pct.csv.gz"
    with gzip.open(full_path, "wt", newline="", encoding="utf-8") as stream:
        full_writer = csv.writer(stream)
        full_writer.writerow(["transaction_id", "is_laundering"])
        for transaction_id, label in con.execute("""
            SELECT transaction_id, is_laundering FROM transactions
            WHERE timestamp_minute < ? ORDER BY transaction_id
        """, (state["train_end_exclusive_minute"],)):
            (positives if label else negatives).append(transaction_id)
            full_writer.writerow([transaction_id, label])
    add_two_percent = math.ceil(len(positives) * 0.01) < 30
    fractions = (0.01, 0.02, 0.05, 0.20) if add_two_percent else (0.01, 0.05, 0.20)
    counts = {}
    with gzip.open(root / "processed" / "label_subsets.csv.gz", "wt", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["seed", "label_fraction", "transaction_id", "is_laundering"])
        for seed in SEEDS:
            positive_ids = positives.copy()
            negative_ids = negatives.copy()
            random.Random(seed).shuffle(positive_ids)
            random.Random(seed + 10000).shuffle(negative_ids)
            for fraction in fractions:
                p_count = max(1, math.ceil(len(positive_ids) * fraction))
                n_count = max(1, math.ceil(len(negative_ids) * fraction))
                for transaction_id in positive_ids[:p_count]:
                    writer.writerow([seed, fraction, transaction_id, 1])
                for transaction_id in negative_ids[:n_count]:
                    writer.writerow([seed, fraction, transaction_id, 0])
                counts[f"{seed}:{fraction}"] = {"positive": p_count, "negative": n_count}
    write_json(root / "processed" / "label_subset_manifest.json", {
        "seeds": SEEDS,
        "fractions_exported": fractions,
        "fraction_1_0": "all train transactions",
        "fraction_1_0_file": full_path.name,
        "fraction_1_0_reused_for_seeds": SEEDS,
        "fraction_1_0_counts": {"positive": len(positives), "negative": len(negatives)},
        "train_positive": len(positives),
        "train_negative": len(negatives),
        "added_2_percent_due_to_under_30_positives_at_1_percent": add_two_percent,
        "counts": counts,
    })
    log(f"Exported stratified subsets for {len(SEEDS)} seeds and one shared 100% train list")

def verify(root: Path, con: sqlite3.Connection) -> None:
    state = load_state(root)
    npy_path = root / "processed" / "event_tokens.npy"
    if npy_path.exists():
        tokens = np.load(npy_path, mmap_mode="r")
    else:
        tokens = np.load(root / "processed" / "event_tokens.npz")["tokens"]
    if tokens.shape != (state["event_count"], len(TOKEN_FIELDS)):
        raise AssertionError("Token matrix shape does not match source counts")
    if np.any(tokens < 0):
        raise AssertionError("Token matrix contains negative IDs")
    if con.execute(
        "SELECT COUNT(*) FROM transactions WHERE typology IS NOT NULL AND is_laundering=0"
    ).fetchone()[0]:
        raise AssertionError("A pattern row matched a negative label")
    summary = json.loads((root / "reports" / "eda_summary.json").read_text(encoding="utf-8"))
    if sum(row["transactions"] for row in summary["split_counts"]) != state["transaction_count"]:
        raise AssertionError("Split counts do not sum to all transactions")
    with gzip.open(root / "processed" / "window_index.csv.gz", "rt", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        sample = [next(reader) for _ in range(1000)]
    for row in sample:
        start = int(row["window_start_index"])
        end = int(row["target_event_index"])
        if end < start or end - start + 1 != int(row["window_length"]):
            raise AssertionError("Window index is inconsistent")
    full_rows = full_positive = 0
    last_id = -1
    with gzip.open(root / "processed" / "label_subset_100pct.csv.gz", "rt", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["transaction_id", "is_laundering"]:
            raise AssertionError("100% train list has unexpected columns")
        for row in reader:
            transaction_id = int(row["transaction_id"])
            label = int(row["is_laundering"])
            if transaction_id <= last_id or label not in (0, 1):
                raise AssertionError("100% train list is not sorted and unique")
            last_id = transaction_id
            full_rows += 1
            full_positive += label
    train = next(row for row in summary["split_counts"] if row["split"] == "train")
    if full_rows != train["transactions"] or full_positive != train["laundering"]:
        raise AssertionError("100% train list does not match train split totals")
    write_json(root / "reports" / "verification.json", {
        "status": "passed",
        "token_shape": list(tokens.shape),
        "split_counts": summary["split_counts"],
        "pattern_matches": state["matched_pattern_row_keys"],
        "sample_windows_checked": len(sample),
        "full_train_list_rows": full_rows,
        "full_train_list_positives": full_positive,
    })
    log("Verification passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("ingest", "eda", "export", "subsets", "verify", "all"),
        default="all",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    for subdir in ("raw", "processed", "reports", "figures"):
        (root / subdir).mkdir(parents=True, exist_ok=True)
    con = connect(args.db.resolve())
    try:
        stages = ("ingest", "eda", "export", "subsets", "verify") if args.stage == "all" else (args.stage,)
        for stage in stages:
            log(f"Starting {stage}")
            globals()[stage](root, con)
    finally:
        con.close()


if __name__ == "__main__":
    main()
