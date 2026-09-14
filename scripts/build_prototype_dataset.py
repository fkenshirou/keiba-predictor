#!/usr/bin/env python3
"""Build compact prototype datasets from the forked keiba-predictor repository.

The script deliberately separates target/outcome columns from pre-race KPT signals.
Final popularity and final win odds are retained only as market/outcome reference columns;
model code must not use them as pre-race features.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup


HORSE_HEADERS = {
    "枠番",
    "馬番",
    "人気",
    "オッズ",
    "馬名",
    "騎手",
    "調教師",
    "予測タイム(sec)",
    "1_順位",
    "1-3着内確率(%)",
    "2_順位",
    "ﾀﾞｰｸﾎｰｽ確率(%)",
    "3_順位",
}


@dataclass
class BuildStats:
    result_source_files: int = 0
    result_rows_raw: int = 0
    result_rows_deduped: int = 0
    result_races: int = 0
    kpt_source_files: int = 0
    kpt_rows_raw: int = 0
    kpt_rows_deduped: int = 0
    kpt_races: int = 0
    kpt_rows_with_race_id: int = 0
    joined_rows: int = 0
    joined_races: int = 0


def clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def strip_html(value: object) -> object:
    if not isinstance(value, str):
        return value
    return BeautifulSoup(value, "html.parser").get_text(" ", strip=True)


def normalize_key(value: object) -> str:
    text = unicodedata.normalize("NFKC", clean_text(value))
    text = text.strip("【】")
    text = re.sub(r"\s*\d{4}/\d{2}/\d{2}\([^)]*\)\s*", "", text)
    text = re.sub(r"\((?:J\.?G)?[ⅠⅡⅢIVX0-9A-Z.]+[^)]*\)", "", text, flags=re.I)
    text = text.replace("ステークス", "S").replace("カップ", "C")
    text = re.sub(r"[\s・\-_.]", "", text)
    return text.upper()


def race_id_from_url(value: object) -> str | None:
    text = clean_text(value)
    for pattern in (r"race_id=(\d{12})", r"id=c?(\d{12})"):
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None


def parse_japanese_date(value: object) -> str | None:
    text = clean_text(value)
    match = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", text)
    if not match:
        return None
    year, month, day = map(int, match.groups())
    return f"{year:04d}-{month:02d}-{day:02d}"


def parse_slash_date(value: object) -> str | None:
    text = clean_text(value)
    match = re.search(r"(\d{4})/(\d{1,2})/(\d{1,2})", text)
    if not match:
        return None
    year, month, day = map(int, match.groups())
    return f"{year:04d}-{month:02d}-{day:02d}"


def parse_distance(value: object) -> tuple[str | None, float | None, str | None]:
    text = clean_text(value)
    surface = None
    if text.startswith("芝"):
        surface = "turf"
    elif text.startswith("ダ"):
        surface = "dirt"
    elif text.startswith("障"):
        surface = "jump"

    match = re.search(r"(\d{3,4})m", text)
    distance = float(match.group(1)) if match else None

    direction = None
    if "左" in text:
        direction = "left"
    elif "右" in text:
        direction = "right"
    elif "直" in text:
        direction = "straight"
    return surface, distance, direction


def parse_body_weight(value: object) -> tuple[float | None, float | None]:
    text = clean_text(value)
    match = re.search(r"(\d+)\s*\(([+-]?\d+|0)\)", text)
    if not match:
        return None, None
    return float(match.group(1)), float(match.group(2))


def parse_numeric_finish(value: object) -> float | None:
    text = clean_text(value)
    if re.fullmatch(r"\d+", text):
        return float(text)
    return None


def race_timestamp(date_iso: str | None, start_time: object) -> str | None:
    if not date_iso:
        return None
    text = clean_text(start_time)
    if not re.fullmatch(r"\d{1,2}:\d{2}", text):
        return None
    try:
        return datetime.strptime(f"{date_iso} {text}", "%Y-%m-%d %H:%M").isoformat()
    except ValueError:
        return None


def load_result_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    rows: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        date_iso = parse_japanese_date(row.get("開催日"))
        surface, distance_m, direction = parse_distance(row.get("距離"))
        body_weight, body_weight_diff = parse_body_weight(row.get("馬体重(増減)"))
        finish = parse_numeric_finish(row.get("着順"))
        race_id = race_id_from_url(row.get("url"))
        rows.append(
            {
                "source_file": path.name,
                "race_id": race_id,
                "race_date": date_iso,
                "race_timestamp": race_timestamp(date_iso, row.get("発走時刻")),
                "venue": clean_text(row.get("開催地")),
                "race_no": pd.to_numeric(row.get("レースNo", "").replace("R", ""), errors="coerce"),
                "race_name": clean_text(row.get("レース名")),
                "surface": surface,
                "distance_m": distance_m,
                "direction": direction,
                "distance_raw": clean_text(row.get("距離")),
                "weather": clean_text(row.get("天候")),
                "going": clean_text(row.get("馬場")),
                "frame_no": pd.to_numeric(row.get("枠"), errors="coerce"),
                "horse_no": pd.to_numeric(row.get("馬番"), errors="coerce"),
                "horse_name": clean_text(row.get("馬名")),
                "horse_name_key": normalize_key(row.get("馬名")),
                "sex_age": clean_text(row.get("性齢")),
                "sex": clean_text(row.get("gender")),
                "age": pd.to_numeric(row.get("age"), errors="coerce"),
                "carried_weight": pd.to_numeric(row.get("斤量"), errors="coerce"),
                "jockey": clean_text(row.get("騎手")),
                "trainer": clean_text(row.get("厩舎")),
                "body_weight": body_weight,
                "body_weight_diff": body_weight_diff,
                "finish_position": finish,
                "finish_position_raw": clean_text(row.get("着順")),
                "race_time_raw": clean_text(row.get("タイム")),
                "margin": clean_text(row.get("着差")),
                "last_3f": pd.to_numeric(row.get("後3F"), errors="coerce"),
                "corner_positions": clean_text(row.get("コーナー通過順")),
                # Market/outcome reference only. Never use these as pre-race features.
                "final_popularity": pd.to_numeric(row.get("人気"), errors="coerce"),
                "final_win_odds": pd.to_numeric(row.get("単勝オッズ"), errors="coerce"),
                "y_win": 1 if finish == 1 else (0 if finish is not None else None),
                "y_top3": 1 if finish is not None and finish <= 3 else (0 if finish is not None else None),
            }
        )
    return pd.DataFrame(rows)


def build_results(csv_dir: Path, stats: BuildStats) -> pd.DataFrame:
    paths = sorted(csv_dir.glob("race_result_*.csv"))
    stats.result_source_files = len(paths)
    frames = [load_result_csv(path) for path in paths]
    if not frames:
        return pd.DataFrame()
    results = pd.concat(frames, ignore_index=True)
    stats.result_rows_raw = len(results)

    # Several source CSVs overlap in date ranges. Prefer the lexicographically latest source file.
    results = results.sort_values(["race_date", "race_id", "horse_name_key", "source_file"])
    results = results.drop_duplicates(["race_id", "horse_name_key"], keep="last")
    results = results.sort_values(["race_timestamp", "race_id", "horse_no"], na_position="last")
    stats.result_rows_deduped = len(results)
    stats.result_races = int(results["race_id"].nunique(dropna=True))
    return results.reset_index(drop=True)


def parse_heading(text: str) -> tuple[str, str | None]:
    stripped = clean_text(text).strip("【】")
    date_iso = parse_slash_date(stripped)
    race_name = re.sub(r"\s+\d{4}/\d{1,2}/\d{1,2}\([^)]*\)\s*$", "", stripped)
    return race_name, date_iso


def parse_datatables(path: Path) -> tuple[list[dict[str, object]], str]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    soup = BeautifulSoup(text, "html.parser")
    tables: list[dict[str, object]] = []
    for script in soup.find_all("script", {"type": "application/json"}):
        content = script.string or script.get_text()
        if not content or '"container":"<table' not in content:
            continue
        try:
            obj = json.loads(content)
        except json.JSONDecodeError:
            continue
        x = obj.get("x", {})
        container = x.get("container", "")
        headers = [
            th.get_text(" ", strip=True)
            for th in BeautifulSoup(container, "html.parser").find_all("th")
        ]
        data = x.get("data", [])
        if not headers or len(data) != len(headers):
            continue
        nrows = max((len(column) for column in data), default=0)
        columns: dict[str, list[object]] = {}
        for header, column in zip(headers, data, strict=False):
            values = [strip_html(value) for value in column]
            values.extend([None] * (nrows - len(values)))
            columns[header] = values
        frame = pd.DataFrame(columns)

        widget_id = script.get("data-for")
        div = soup.find(id=widget_id) if widget_id else None
        h3_node = div.find_previous("h3") if div else None
        h2_node = div.find_previous("h2") if div else None
        tables.append(
            {
                "h2": h2_node.get_text(" ", strip=True) if h2_node else "",
                "h3": h3_node.get_text(" ", strip=True) if h3_node else "",
                "headers": headers,
                "df": frame,
            }
        )
    return tables, text


def best_race_id(race_name: str, race_date: str | None, race_info: pd.DataFrame) -> str | None:
    if race_info.empty or not race_date:
        return None
    date_candidates = []
    for _, row in race_info.iterrows():
        info_date = parse_slash_date(row.get("日付")) or parse_japanese_date(row.get("日付"))
        if info_date == race_date:
            date_candidates.append(row)
    if not date_candidates:
        return None

    target = normalize_key(race_name)
    best_score = -1.0
    best_url = None
    for row in date_candidates:
        candidate = normalize_key(row.get("レース名", ""))
        score = SequenceMatcher(None, target, candidate).ratio()
        if target and candidate and (target in candidate or candidate in target):
            score = max(score, 0.95)
        if score > best_score:
            best_score = score
            best_url = row.get("URL(netkeiba)")
    return race_id_from_url(best_url) if best_score >= 0.55 else None


def parse_kpt_report(path: Path) -> pd.DataFrame:
    tables, _ = parse_datatables(path)
    race_info_frames = [table["df"] for table in tables if "直近の全レース情報" in table["h2"]]
    race_info = pd.concat(race_info_frames, ignore_index=True) if race_info_frames else pd.DataFrame()

    rows: list[dict[str, object]] = []
    for table in tables:
        headers = set(table["headers"])
        if not HORSE_HEADERS.issubset(headers) or "推し馬" in headers:
            continue
        race_name, race_date = parse_heading(table["h3"])
        race_id = best_race_id(race_name, race_date, race_info)
        for _, row in table["df"].iterrows():
            rows.append(
                {
                    "source_file": path.name,
                    "race_id": race_id,
                    "race_date": race_date,
                    "race_name": race_name,
                    "race_name_key": normalize_key(race_name),
                    "frame_no": pd.to_numeric(row.get("枠番"), errors="coerce"),
                    "horse_no": pd.to_numeric(row.get("馬番"), errors="coerce"),
                    "horse_name": clean_text(row.get("馬名")),
                    "horse_name_key": normalize_key(row.get("馬名")),
                    "jockey": clean_text(row.get("騎手")),
                    "trainer": clean_text(row.get("調教師")),
                    # KPT pre-race model outputs.
                    "kpt_predicted_time_sec": pd.to_numeric(
                        row.get("予測タイム(sec)"), errors="coerce"
                    ),
                    "kpt_predicted_time_rank": pd.to_numeric(row.get("1_順位"), errors="coerce"),
                    "kpt_top3_score_pct": pd.to_numeric(
                        row.get("1-3着内確率(%)"), errors="coerce"
                    ),
                    "kpt_top3_rank": pd.to_numeric(row.get("2_順位"), errors="coerce"),
                    "kpt_darkhorse_score_pct": pd.to_numeric(
                        row.get("ﾀﾞｰｸﾎｰｽ確率(%)"), errors="coerce"
                    ),
                    "kpt_darkhorse_rank": pd.to_numeric(row.get("3_順位"), errors="coerce"),
                    # The report's popularity/odds semantics vary by era. Keep reference-only.
                    "kpt_report_popularity": pd.to_numeric(row.get("人気"), errors="coerce"),
                    "kpt_report_win_odds": pd.to_numeric(row.get("オッズ"), errors="coerce"),
                }
            )
    return pd.DataFrame(rows)


def build_kpt(report_dir: Path, stats: BuildStats) -> pd.DataFrame:
    paths = sorted(report_dir.glob("PredResult*.html"))
    dated_paths = [path for path in paths if path.name != "PredResult_latest_race.html"]
    paths = dated_paths or paths
    stats.kpt_source_files = len(paths)

    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = parse_kpt_report(path)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()

    kpt = pd.concat(frames, ignore_index=True)
    stats.kpt_rows_raw = len(kpt)
    kpt["dedupe_race_key"] = kpt["race_id"].fillna(
        kpt["race_date"].fillna("") + "|" + kpt["race_name_key"].fillna("")
    )
    kpt = kpt.sort_values(["race_date", "dedupe_race_key", "horse_name_key", "source_file"])
    kpt = kpt.drop_duplicates(["dedupe_race_key", "horse_name_key"], keep="last")
    stats.kpt_rows_deduped = len(kpt)
    stats.kpt_rows_with_race_id = int(kpt["race_id"].notna().sum())
    stats.kpt_races = int(kpt["dedupe_race_key"].nunique(dropna=True))
    return kpt.drop(columns=["dedupe_race_key"]).reset_index(drop=True)


def join_kpt_results(kpt: pd.DataFrame, results: pd.DataFrame, stats: BuildStats) -> pd.DataFrame:
    if kpt.empty or results.empty:
        return pd.DataFrame()
    kpt_with_id = kpt[kpt["race_id"].notna()].copy()
    result_cols = [
        "race_id",
        "horse_name_key",
        "race_timestamp",
        "venue",
        "surface",
        "distance_m",
        "direction",
        "weather",
        "going",
        "carried_weight",
        "body_weight",
        "body_weight_diff",
        "finish_position",
        "y_win",
        "y_top3",
        "final_popularity",
        "final_win_odds",
        "last_3f",
        "corner_positions",
    ]
    joined = kpt_with_id.merge(
        results[result_cols],
        on=["race_id", "horse_name_key"],
        how="inner",
        validate="many_to_one",
    )
    stats.joined_rows = len(joined)
    stats.joined_races = int(joined["race_id"].nunique(dropna=True))
    return joined.sort_values(["race_timestamp", "race_id", "horse_no"]).reset_index(drop=True)


def write_frame(frame: pd.DataFrame, stem: Path) -> None:
    frame.to_csv(stem.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    frame.to_parquet(stem.with_suffix(".parquet"), index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("derived/prototype"))
    args = parser.parse_args()

    root = args.repo_root.resolve()
    output = (root / args.output).resolve() if not args.output.is_absolute() else args.output
    output.mkdir(parents=True, exist_ok=True)

    stats = BuildStats()
    results = build_results(root / "csv", stats)
    kpt = build_kpt(root / "report", stats)
    joined = join_kpt_results(kpt, results, stats)

    write_frame(results, output / "race_results")
    write_frame(kpt, output / "kpt_predictions")
    write_frame(joined, output / "kpt_results_joined")

    manifest = {
        "generated_at_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "stats": asdict(stats),
        "result_date_min": results["race_date"].min() if not results.empty else None,
        "result_date_max": results["race_date"].max() if not results.empty else None,
        "kpt_date_min": kpt["race_date"].min() if not kpt.empty else None,
        "kpt_date_max": kpt["race_date"].max() if not kpt.empty else None,
        "notes": [
            "final_popularity/final_win_odds are market reference columns only, never pre-race features",
            "finish_position/last_3f/corner_positions are target-race outcomes and never pre-race features",
            "KPT report odds/popularity semantics vary by era and are reference-only",
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
