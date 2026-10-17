"""Pinned public data download and complete-user sampling. No credentials needed."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
REVISION = "f69775be31fa5779907cf0a92ddedb70037fb5ae"
BASE = f"https://raw.githubusercontent.com/irsafilo/KION_DATASET/{REVISION}"
SOURCES = {"data_en.zip": f"{BASE}/data_en.zip", "items.csv": f"{BASE}/data_original/items.csv"}


def dump(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download():
    raw = ROOT / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    expected_file = ROOT / "checksums.json"
    expected = json.loads(expected_file.read_text()) if expected_file.exists() else {}
    for name, url in SOURCES.items():
        path = raw / name
        if not path.exists():
            tmp = path.with_suffix(path.suffix + ".part")
            with requests.get(url, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                with tmp.open("wb") as out:
                    for chunk in response.iter_content(1024 * 1024):
                        out.write(chunk)
            tmp.replace(path)
        if name in expected and sha256(path) != expected[name]:
            raise ValueError(f"Checksum mismatch: {path}")


def read_events():
    with zipfile.ZipFile(ROOT / "data/raw/data_en.zip") as z:
        assert z.testzip() is None, "Archive CRC error"
        members = [n for n in z.namelist() if n.endswith('/interactions.csv') or n == 'interactions.csv']
        assert len(members) == 1
        with z.open(members[0]) as stream:
            events = pd.read_csv(stream, usecols=["user_id", "item_id", "last_watch_dt"])
    # Hard failure rather than silently using a truncated source.
    assert len(events) == 5_476_251, f"Incomplete KION source: {len(events)} rows"
    assert events.last_watch_dt.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all()
    events["timestamp"] = pd.to_datetime(events.pop("last_watch_dt"), format="%Y-%m-%d")
    assert not events.isna().any().any()
    assert not events.duplicated(["user_id", "item_id"]).any()
    return events


def sample_users(events, n_users, seed, first_cutoff):
    eligible = np.sort(events.loc[events.timestamp < first_cutoff, "user_id"].unique())
    ids = np.sort(np.random.default_rng(seed).choice(eligible, min(n_users, len(eligible)), replace=False))
    sampled = events[events.user_id.isin(ids)].copy()
    # Include every source row, even future rows, in the sampled raw dataset.
    # The feature pipeline later applies the temporal boundary.
    full_counts = events.groupby("user_id").size().reindex(ids)
    pd.testing.assert_series_equal(full_counts, sampled.groupby("user_id").size().reindex(ids))
    return sampled.sort_values(["timestamp", "user_id", "item_id"]).reset_index(drop=True), ids


def prepare(n_users=10000, seed=42):
    download()
    events = read_events()
    end = events.timestamp.max() + pd.Timedelta(days=1)
    cutoffs = [end - pd.Timedelta(days=7 * k) for k in range(6, 0, -1)]
    sampled, ids = sample_users(events, n_users, seed, cutoffs[0])
    path = ROOT / f"data/cohort_{n_users}"
    path.mkdir(parents=True, exist_ok=True)
    sampled.to_parquet(path / "events.parquet", index=False)
    pd.Series(ids, name="user_id").to_csv(path / "user_ids.csv", index=False)
    items = pd.read_csv(ROOT / "data/raw/items.csv")
    assert not items.item_id.duplicated().any()
    items.to_parquet(path / "items.parquet", index=False)
    manifest = dict(source_revision=REVISION, source_urls=SOURCES,
                    checksums={n: sha256(ROOT / 'data/raw' / n) for n in SOURCES},
                    population_rows=len(events), population_users=events.user_id.nunique(),
                    population_items=events.item_id.nunique(), source_start=str(events.timestamp.min()),
                    source_end=str(events.timestamp.max()), seed=seed, n_users=len(ids),
                    rows=len(sampled), items=sampled.item_id.nunique(),
                    full_user_histories_verified=True,
                    cutoffs={name: str(t.date()) for name,t in zip(
                        ['train_1','train_2','train_3','search','tune','test'],cutoffs)},
                    end=str(end.date()), cohort_sha256=sha256(path / 'user_ids.csv'))
    dump(path / "manifest.json", manifest)
    dump(ROOT / "results" / f"data_manifest_{n_users}.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.users, args.seed)
