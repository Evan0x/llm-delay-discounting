"""Plotting utilities for delay-discounting experiment outputs."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import _DELAY_DAYS, exp_model, hyp_model

LOGGER = logging.getLogger(__name__)


def _save_figure(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(f"{out_base}.png", dpi=200)
    fig.savefig(f"{out_base}.pdf")
    plt.close(fig)


def plot_choice_curves(parsed: pd.DataFrame, out_dir: Path) -> None:
    """Plot choice probabilities over immediate amount per beneficiary and delay."""
    if parsed.empty:
        return

    usable = parsed.loc[(~parsed["parse_fail"]) & (parsed["chose_immediate"].notna())].copy()
    if usable.empty:
        return

    for model_name, df_model in usable.groupby("model_name"):
        beneficiaries = sorted(df_model["beneficiary_key"].unique())
        nrows = len(beneficiaries)
        fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(8, 3 * nrows), sharex=True)
        if nrows == 1:
            axes = [axes]

        for ax, beneficiary in zip(axes, beneficiaries):
            df_b = df_model[df_model["beneficiary_key"] == beneficiary]
            grouped = (
                df_b.groupby(["delay", "immediate_amount"], as_index=False)["chose_immediate"]
                .mean()
                .sort_values("immediate_amount")
            )
            for delay, df_d in grouped.groupby("delay"):
                ax.plot(
                    df_d["immediate_amount"],
                    df_d["chose_immediate"],
                    marker="o",
                    label=delay,
                )
            ax.set_title(f"{model_name} | beneficiary={beneficiary}")
            ax.set_ylabel("P(choose immediate)")
            ax.set_ylim(-0.05, 1.05)
            ax.legend(title="Delay", fontsize=8)

        axes[-1].set_xlabel("Immediate amount")
        _save_figure(fig, out_dir / f"choice_curves_{model_name}")


def plot_indifference_with_fits(indiff: pd.DataFrame, model_fits: pd.DataFrame, out_dir: Path) -> None:
    """Plot indifference points vs delay with exponential/hyperbolic fits."""
    if indiff.empty or model_fits.empty:
        return

    for (model_name, beneficiary), df_points in indiff.groupby(["model_name", "beneficiary_key"]):
        fig, ax = plt.subplots(figsize=(7, 5))
        uncensored = df_points[(~df_points["censored"]) & df_points["indifference_point"].notna()]

        if not uncensored.empty:
            ax.scatter(uncensored["delay_days"], uncensored["indifference_point"], label="Observed", zorder=3)

            x_grid = np.linspace(min(_DELAY_DAYS.values()), max(_DELAY_DAYS.values()), 200)
            for model_type, model_fn, style in [
                ("exponential", exp_model, "-"),
                ("hyperbolic", hyp_model, "--"),
            ]:
                fit_row = model_fits[
                    (model_fits["model_name"] == model_name)
                    & (model_fits["beneficiary_key"] == beneficiary)
                    & (model_fits["discount_model"] == model_type)
                ]
                if fit_row.empty or pd.isna(fit_row.iloc[0]["k"]):
                    continue
                k = float(fit_row.iloc[0]["k"])
                y = model_fn(x_grid, k)
                ax.plot(x_grid, y, style, label=f"{model_type} (k={k:.4g})")

        ax.set_title(f"{model_name} | {beneficiary}")
        ax.set_xlabel("Delay (days)")
        ax.set_ylabel("Indifference point")
        ax.set_ylim(0, 105)
        ax.legend()
        _save_figure(fig, out_dir / f"indifference_fits_{model_name}_{beneficiary}")


def plot_k_forest(k_boot: pd.DataFrame, out_dir: Path) -> None:
    """Forest-style plot of k with bootstrap confidence intervals."""
    if k_boot.empty:
        return

    for (model_name, discount_model), df_group in k_boot.groupby(["model_name", "discount_model"]):
        dfp = df_group.sort_values("beneficiary_key").reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(8, 4 + 0.35 * len(dfp)))

        y = np.arange(len(dfp))
        x = dfp["k_mean"].to_numpy(dtype=float)
        low = dfp["k_ci_low"].to_numpy(dtype=float)
        high = dfp["k_ci_high"].to_numpy(dtype=float)

        xerr = np.vstack([x - low, high - x])
        ax.errorbar(x, y, xerr=xerr, fmt="o", capsize=4)
        ax.set_yticks(y)
        ax.set_yticklabels(dfp["beneficiary_key"].tolist())
        ax.set_xlabel("k")
        ax.set_title(f"{model_name} | {discount_model} k estimates")
        ax.axvline(0, color="gray", linewidth=1)

        _save_figure(fig, out_dir / f"k_forest_{model_name}_{discount_model}")


def main() -> None:
    """CLI for plotting all figures."""
    parser = argparse.ArgumentParser(description="Generate experiment figures")
    parser.add_argument("--parsed", default="data/processed/parsed_trials.csv")
    parser.add_argument("--indiff", default="results/tables/indifference_points.csv")
    parser.add_argument("--fits", default="results/tables/discount_model_fits.csv")
    parser.add_argument("--k-bootstrap", default="results/tables/k_bootstrap.csv")
    parser.add_argument("--out-dir", default="results/figures")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    parsed = pd.read_csv(args.parsed) if Path(args.parsed).exists() else pd.DataFrame()
    indiff = pd.read_csv(args.indiff) if Path(args.indiff).exists() else pd.DataFrame()
    fits = pd.read_csv(args.fits) if Path(args.fits).exists() else pd.DataFrame()
    k_boot = pd.read_csv(args.k_bootstrap) if Path(args.k_bootstrap).exists() else pd.DataFrame()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_choice_curves(parsed, out_dir)
    plot_indifference_with_fits(indiff, fits, out_dir)
    plot_k_forest(k_boot, out_dir)

    LOGGER.info("Figures saved to %s", out_dir)


if __name__ == "__main__":
    main()
