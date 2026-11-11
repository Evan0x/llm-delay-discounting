"""Analysis for LLM delay-discounting experiments."""

from __future__ import annotations

import argparse
import itertools
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import statsmodels.api as sm

from parse import parse_fail_report, parse_jsonl_files

LOGGER = logging.getLogger(__name__)


_DELAY_DAYS = {
    "1 day": 1.0,
    "1 week": 7.0,
    "1 month": 30.0,
    "6 months": 182.5,
    "1 year": 365.0,
    "5 years": 1825.0,
}


def delay_to_days(delay: str) -> float:
    """Convert delay label into days."""
    if delay not in _DELAY_DAYS:
        raise ValueError(f"Unknown delay label: {delay}")
    return _DELAY_DAYS[delay]


def fit_indifference_for_group(df_group: pd.DataFrame) -> dict[str, Any]:
    """Fit logistic choice curve and estimate indifference point."""
    y = df_group["chose_immediate"].astype(int)
    x = df_group["immediate_amount"].astype(float)

    result: dict[str, Any] = {
        "n": len(df_group),
        "all_immediate": bool((y == 1).all()),
        "all_delayed": bool((y == 0).all()),
        "censored": False,
        "censoring_side": None,
        "indifference_point": np.nan,
        "beta0": np.nan,
        "beta1": np.nan,
    }

    if result["all_immediate"]:
        result["censored"] = True
        result["censoring_side"] = "high"  # true threshold above tested range
        return result

    if result["all_delayed"]:
        result["censored"] = True
        result["censoring_side"] = "low"  # true threshold below tested range
        return result

    X = sm.add_constant(x)
    model = sm.Logit(y, X)
    fitted = model.fit(disp=False)

    beta0 = float(fitted.params.iloc[0])
    beta1 = float(fitted.params.iloc[1])
    indiff = np.nan if np.isclose(beta1, 0.0) else -beta0 / beta1

    result.update(
        {
            "beta0": beta0,
            "beta1": beta1,
            "indifference_point": float(indiff),
        }
    )
    return result


def estimate_indifference_points(parsed_df: pd.DataFrame) -> pd.DataFrame:
    """Estimate indifference points per (model, beneficiary, delay)."""
    usable = parsed_df.loc[~parsed_df["parse_fail"].astype(bool)].copy()
    usable = usable.loc[usable["chose_immediate"].notna()].copy()

    rows: list[dict[str, Any]] = []
    group_keys = ["model_name", "beneficiary_key", "delay"]
    for key, df_group in usable.groupby(group_keys):
        fit = fit_indifference_for_group(df_group)
        row = {
            "model_name": key[0],
            "beneficiary_key": key[1],
            "delay": key[2],
            "delay_days": delay_to_days(key[2]),
            **fit,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def exp_model(days: np.ndarray, k: float) -> np.ndarray:
    """Exponential discounting model."""
    return 100.0 * np.exp(-k * days)


def hyp_model(days: np.ndarray, k: float) -> np.ndarray:
    """Hyperbolic discounting model."""
    return 100.0 / (1.0 + k * days)


def fit_discount_curve(df_points: pd.DataFrame, model_type: str) -> dict[str, Any]:
    """Fit k for exponential or hyperbolic model."""
    model_fn = exp_model if model_type == "exponential" else hyp_model
    clean = df_points.loc[(~df_points["censored"]) & df_points["indifference_point"].notna()].copy()
    clean = clean.loc[(clean["indifference_point"] > 0) & (clean["indifference_point"] <= 100)]

    if len(clean) < 3:
        return {
            "k": np.nan,
            "n": len(clean),
            "rss": np.nan,
            "aic": np.nan,
            "bic": np.nan,
        }

    x = clean["delay_days"].to_numpy(dtype=float)
    y = clean["indifference_point"].to_numpy(dtype=float)

    try:
        popt, _ = curve_fit(model_fn, x, y, p0=[0.01], bounds=(1e-8, 10.0), maxfev=10000)
        k = float(popt[0])
    except Exception:  # noqa: BLE001
        return {
            "k": np.nan,
            "n": len(clean),
            "rss": np.nan,
            "aic": np.nan,
            "bic": np.nan,
        }

    yhat = model_fn(x, k)
    rss = float(np.sum((y - yhat) ** 2))
    n = len(y)
    p = 1
    eps = 1e-12
    aic = n * np.log((rss + eps) / n) + 2 * p
    bic = n * np.log((rss + eps) / n) + p * np.log(n)

    return {"k": k, "n": n, "rss": rss, "aic": aic, "bic": bic}


def fit_discount_models(indiff_df: pd.DataFrame) -> pd.DataFrame:
    """Fit exponential and hyperbolic models per (model, beneficiary)."""
    rows: list[dict[str, Any]] = []
    for (model_name, beneficiary_key), df_group in indiff_df.groupby(["model_name", "beneficiary_key"]):
        for model_type in ("exponential", "hyperbolic"):
            fit = fit_discount_curve(df_group, model_type=model_type)
            rows.append(
                {
                    "model_name": model_name,
                    "beneficiary_key": beneficiary_key,
                    "discount_model": model_type,
                    **fit,
                }
            )
    return pd.DataFrame(rows)


def bootstrap_k(
    parsed_df: pd.DataFrame,
    discount_model: str,
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    """Bootstrap k by beneficiary/model from trial-level resampling."""
    usable = parsed_df.loc[~parsed_df["parse_fail"].astype(bool)].copy()
    usable = usable.loc[usable["chose_immediate"].notna()].copy()

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []

    for (model_name, beneficiary_key), df_group in usable.groupby(["model_name", "beneficiary_key"]):
        n = len(df_group)
        if n == 0:
            continue

        samples: list[float] = []
        for _ in range(iterations):
            idx = rng.integers(0, n, size=n)
            sample = df_group.iloc[idx]
            indiff = estimate_indifference_points(sample)
            subset = indiff[
                (indiff["model_name"] == model_name)
                & (indiff["beneficiary_key"] == beneficiary_key)
            ]
            fit = fit_discount_curve(subset, model_type=discount_model)
            samples.append(float(fit["k"]) if pd.notna(fit["k"]) else np.nan)

        arr = np.array(samples, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            rows.append(
                {
                    "model_name": model_name,
                    "beneficiary_key": beneficiary_key,
                    "discount_model": discount_model,
                    "k_mean": np.nan,
                    "k_ci_low": np.nan,
                    "k_ci_high": np.nan,
                    "n_boot": 0,
                }
            )
            continue

        rows.append(
            {
                "model_name": model_name,
                "beneficiary_key": beneficiary_key,
                "discount_model": discount_model,
                "k_mean": float(np.mean(arr)),
                "k_ci_low": float(np.percentile(arr, 2.5)),
                "k_ci_high": float(np.percentile(arr, 97.5)),
                "n_boot": int(arr.size),
            }
        )

    return pd.DataFrame(rows)


def bootstrap_k_differences(
    parsed_df: pd.DataFrame,
    discount_model: str,
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    """Bootstrap pairwise k differences with Bonferroni-corrected CIs."""
    usable = parsed_df.loc[~parsed_df["parse_fail"].astype(bool)].copy()
    usable = usable.loc[usable["chose_immediate"].notna()].copy()
    rng = np.random.default_rng(seed)

    rows: list[dict[str, Any]] = []

    for model_name, df_model in usable.groupby("model_name"):
        beneficiaries = sorted(df_model["beneficiary_key"].unique().tolist())
        pairs = list(itertools.combinations(beneficiaries, 2))
        m = max(1, len(pairs))
        alpha_adj = 0.05 / m
        low_q = 100 * (alpha_adj / 2)
        high_q = 100 * (1 - alpha_adj / 2)

        boot_by_beneficiary: dict[str, list[float]] = {b: [] for b in beneficiaries}

        for _ in range(iterations):
            k_values: dict[str, float] = {}
            for b in beneficiaries:
                df_b = df_model.loc[df_model["beneficiary_key"] == b]
                n = len(df_b)
                if n == 0:
                    k_values[b] = np.nan
                    continue
                idx = rng.integers(0, n, size=n)
                sample = df_b.iloc[idx]
                indiff = estimate_indifference_points(sample)
                subset = indiff[(indiff["model_name"] == model_name) & (indiff["beneficiary_key"] == b)]
                fit = fit_discount_curve(subset, model_type=discount_model)
                k_values[b] = float(fit["k"]) if pd.notna(fit["k"]) else np.nan

            for b, k in k_values.items():
                boot_by_beneficiary[b].append(k)

        for b1, b2 in pairs:
            arr1 = np.array(boot_by_beneficiary[b1], dtype=float)
            arr2 = np.array(boot_by_beneficiary[b2], dtype=float)
            diff = arr1 - arr2
            diff = diff[~np.isnan(diff)]
            if diff.size == 0:
                rows.append(
                    {
                        "model_name": model_name,
                        "discount_model": discount_model,
                        "beneficiary_1": b1,
                        "beneficiary_2": b2,
                        "k_diff_mean": np.nan,
                        "k_diff_ci_low": np.nan,
                        "k_diff_ci_high": np.nan,
                        "n_boot": 0,
                        "correction": "bonferroni",
                        "alpha_adjusted": alpha_adj,
                    }
                )
                continue

            rows.append(
                {
                    "model_name": model_name,
                    "discount_model": discount_model,
                    "beneficiary_1": b1,
                    "beneficiary_2": b2,
                    "k_diff_mean": float(np.mean(diff)),
                    "k_diff_ci_low": float(np.percentile(diff, low_q)),
                    "k_diff_ci_high": float(np.percentile(diff, high_q)),
                    "n_boot": int(diff.size),
                    "correction": "bonferroni",
                    "alpha_adjusted": alpha_adj,
                }
            )

    return pd.DataFrame(rows)


def save_markdown_summary(
    out_path: Path,
    parse_report: pd.DataFrame,
    model_fits: pd.DataFrame,
    k_boot: pd.DataFrame,
) -> None:
    """Write short markdown summary."""
    lines: list[str] = ["# Analysis Summary", ""]

    if parse_report.empty:
        lines.extend(["No parsed data found.", ""])
    else:
        overall_parse_fail = float(parse_report["parse_fail_rate"].mean())
        lines.extend(
            [
                f"- Mean parse-fail rate across conditions: {overall_parse_fail:.3f}",
                f"- Conditions in parse report: {len(parse_report)}",
                "",
            ]
        )

    if not model_fits.empty:
        lines.append("## Model comparison (AIC/BIC)")
        lines.append("")
        best = (
            model_fits.sort_values(["model_name", "beneficiary_key", "aic"]) \
            .groupby(["model_name", "beneficiary_key"], as_index=False) \
            .first()
        )
        for _, row in best.iterrows():
            lines.append(
                f"- {row['model_name']} / {row['beneficiary_key']}: "
                f"best={row['discount_model']} (AIC={row['aic']:.2f}, BIC={row['bic']:.2f})"
            )
        lines.append("")

    if not k_boot.empty:
        lines.append("## Bootstrap k (95% CI)")
        lines.append("")
        for _, row in k_boot.iterrows():
            lines.append(
                f"- {row['model_name']} / {row['beneficiary_key']} / {row['discount_model']}: "
                f"k={row['k_mean']:.6f} [{row['k_ci_low']:.6f}, {row['k_ci_high']:.6f}]"
            )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """CLI for analysis pipeline."""
    parser = argparse.ArgumentParser(description="Analyze delay-discounting experiment")
    parser.add_argument("--raw-glob", default="data/raw/*.jsonl")
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    parsed = parse_jsonl_files(args.raw_glob)
    if parsed.empty:
        LOGGER.warning("No raw data found with glob: %s", args.raw_glob)
        return

    processed_dir = Path("data/processed")
    tables_dir = Path("results/tables")
    processed_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    parsed.to_csv(processed_dir / "parsed_trials.csv", index=False)

    parse_report = parse_fail_report(parsed)
    parse_report.to_csv(tables_dir / "parse_fail_report.csv", index=False)

    indiff = estimate_indifference_points(parsed)
    indiff.to_csv(tables_dir / "indifference_points.csv", index=False)

    model_fits = fit_discount_models(indiff)
    model_fits.to_csv(tables_dir / "discount_model_fits.csv", index=False)

    k_boot_frames = []
    diff_frames = []
    for discount_model in ("exponential", "hyperbolic"):
        k_boot = bootstrap_k(parsed, discount_model, args.bootstrap_iterations, args.seed)
        k_boot_frames.append(k_boot)

        k_diff = bootstrap_k_differences(parsed, discount_model, args.bootstrap_iterations, args.seed)
        diff_frames.append(k_diff)

    k_boot_all = pd.concat(k_boot_frames, ignore_index=True) if k_boot_frames else pd.DataFrame()
    k_boot_all.to_csv(tables_dir / "k_bootstrap.csv", index=False)

    k_diff_all = pd.concat(diff_frames, ignore_index=True) if diff_frames else pd.DataFrame()
    k_diff_all.to_csv(tables_dir / "k_differences_bootstrap.csv", index=False)

    save_markdown_summary(
        out_path=tables_dir / "summary.md",
        parse_report=parse_report,
        model_fits=model_fits,
        k_boot=k_boot_all,
    )

    LOGGER.info("Analysis complete. Outputs written to %s", tables_dir)


if __name__ == "__main__":
    main()
