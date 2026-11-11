"""Parse raw model responses into binary choices."""

from __future__ import annotations

import argparse
import glob
import json
import logging
from pathlib import Path
import re
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)

_LEADING_PUNCT_RE = re.compile(r"^[\s\"'`\(\[\{\<\-:;,\.\!\?]+")


def parse_ab(raw_text: str | None) -> tuple[str | None, bool]:
    """Parse first valid A/B token at response start.

    Accept only if the first non-space/punctuation character is A or B.
    """
    if raw_text is None:
        return None, True

    text = raw_text.strip()
    if not text:
        return None, True

    cleaned = _LEADING_PUNCT_RE.sub("", text)
    if not cleaned:
        return None, True

    first = cleaned[0].upper()
    if first not in {"A", "B"}:
        return None, True

    return first, False


def parse_row(row: dict[str, Any]) -> dict[str, Any]:
    """Parse a single raw record and derive `chose_immediate`."""
    choice, parse_fail = parse_ab(row.get("raw_text"))
    immediate_first = bool(row.get("immediate_first"))

    chose_immediate: bool | None
    if parse_fail or choice is None:
        chose_immediate = None
    else:
        chose_immediate = (choice == "A" and immediate_first) or (choice == "B" and not immediate_first)

    parsed = dict(row)
    parsed.update(
        {
            "choice": choice,
            "parse_fail": parse_fail,
            "chose_immediate": chose_immediate,
        }
    )
    return parsed


def parse_jsonl_files(raw_glob: str = "data/raw/*.jsonl") -> pd.DataFrame:
    """Parse all raw JSONL files into one DataFrame."""
    rows: list[dict[str, Any]] = []
    for path in glob.glob(raw_glob):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rows.append(parse_row(json.loads(line)))

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def parse_fail_report(df: pd.DataFrame) -> pd.DataFrame:
    """Compute parse-fail rate per condition."""
    if df.empty:
        return pd.DataFrame()

    keys = ["model_name", "beneficiary_key", "delay", "variant"]
    report = (
        df.groupby(keys, dropna=False)
        .agg(
            n=("parse_fail", "size"),
            parse_fail_rate=("parse_fail", "mean"),
        )
        .reset_index()
    )
    return report


def main() -> None:
    """CLI for parsing raw responses."""
    parser = argparse.ArgumentParser(description="Parse raw JSONL responses")
    parser.add_argument("--raw-glob", default="data/raw/*.jsonl")
    parser.add_argument("--out", default="data/processed/parsed_trials.csv")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    df = parse_jsonl_files(args.raw_glob)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    report = parse_fail_report(df)
    report_path = out_path.parent / "parse_fail_report.csv"
    report.to_csv(report_path, index=False)

    LOGGER.info("Saved parsed trials: %s rows to %s", len(df), out_path)
    LOGGER.info("Saved parse-fail report to %s", report_path)


if __name__ == "__main__":
    main()
