"""Prompt and trial generation for delay discounting experiments."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import random
from typing import Any


@dataclass(frozen=True)
class Trial:
    """Single experimental trial."""

    trial_id: str
    beneficiary_key: str
    beneficiary_long: str
    beneficiary_receives: str
    delay: str
    immediate_amount: int
    delayed_amount: int
    immediate_first: bool
    repetition: int
    variant: str
    prompt: str
    option_a: str
    option_b: str

    def to_dict(self) -> dict[str, Any]:
        """Convert trial to a serializable dictionary."""
        return asdict(self)


def _format_option(receives: str, amount: int, delay: str | None, use_points: bool) -> str:
    """Format an option text."""
    unit = "points" if use_points else ""
    amount_text = f"{amount} {unit}".strip() if use_points else f"${amount}"
    if delay is None:
        return f"{receives} {amount_text} today."
    return f"{receives} {amount_text} in {delay}."


def build_prompt(
    beneficiary_long: str,
    option_a: str,
    option_b: str,
    instruction_template: str,
) -> str:
    """Build a prompt using the required base template."""
    return (
        f"{instruction_template.format(beneficiary_long=beneficiary_long)}\n"
        "Choose exactly one option.\n\n"
        f"Option A: {option_a}\n"
        f"Option B: {option_b}\n\n"
        "Answer with only the letter A or B."
    )


def build_control_prompt(option_a: str, option_b: str, instruction: str) -> str:
    """Build a control prompt with no beneficiary framing."""
    return (
        f"{instruction}\n"
        "Choose exactly one option.\n\n"
        f"Option A: {option_a}\n"
        f"Option B: {option_b}\n\n"
        "Answer with only the letter A or B."
    )


def generate_trials(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate all trials with deterministic IDs and seeded shuffle."""
    delays: list[str] = config["delays"]
    immediate_amounts: list[int] = config["immediate_amounts"]
    delayed_amount: int = int(config["delayed_amount"])
    repetitions: int = int(config["repetitions"])
    seed: int = int(config["seed"])

    beneficiaries_cfg: dict[str, dict[str, str]] = config["beneficiaries"]
    control_cfg: dict[str, Any] = config.get("control_condition", {})
    variants_cfg: dict[str, dict[str, Any]] = config["prompt_variants"]

    beneficiaries: dict[str, dict[str, str]] = dict(beneficiaries_cfg)
    if control_cfg.get("enabled", False):
        beneficiaries[control_cfg.get("key", "control")] = {
            "long": "",
            "receives": control_cfg.get("receives", "You receive"),
        }

    trials: list[Trial] = []
    for beneficiary_key, beneficiary_values in beneficiaries.items():
        for delay in delays:
            for amount in immediate_amounts:
                for rep in range(1, repetitions + 1):
                    for variant_name, variant_cfg in variants_cfg.items():
                        for immediate_first in (True, False):
                            receives = beneficiary_values["receives"]
                            use_points = bool(variant_cfg.get("use_points", False))

                            immediate_text = _format_option(receives, amount, None, use_points)
                            delayed_text = _format_option(
                                receives,
                                delayed_amount,
                                delay,
                                use_points,
                            )

                            if immediate_first:
                                option_a, option_b = immediate_text, delayed_text
                            else:
                                option_a, option_b = delayed_text, immediate_text

                            if beneficiary_key == control_cfg.get("key", "control"):
                                prompt = build_control_prompt(
                                    option_a=option_a,
                                    option_b=option_b,
                                    instruction=control_cfg.get(
                                        "instruction",
                                        "Choose the better option.",
                                    ),
                                )
                                beneficiary_long = "control"
                            else:
                                beneficiary_long = beneficiary_values["long"]
                                prompt = build_prompt(
                                    beneficiary_long=beneficiary_long,
                                    option_a=option_a,
                                    option_b=option_b,
                                    instruction_template=variant_cfg["instruction_template"],
                                )

                            trial_id = (
                                f"{beneficiary_key}|{delay}|{amount}|"
                                f"{str(immediate_first).lower()}|{rep}|{variant_name}"
                            )

                            trials.append(
                                Trial(
                                    trial_id=trial_id,
                                    beneficiary_key=beneficiary_key,
                                    beneficiary_long=beneficiary_long,
                                    beneficiary_receives=receives,
                                    delay=delay,
                                    immediate_amount=amount,
                                    delayed_amount=delayed_amount,
                                    immediate_first=immediate_first,
                                    repetition=rep,
                                    variant=variant_name,
                                    prompt=prompt,
                                    option_a=option_a,
                                    option_b=option_b,
                                )
                            )

    rng = random.Random(seed)
    rng.shuffle(trials)
    return [trial.to_dict() for trial in trials]
