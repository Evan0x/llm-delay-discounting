# LLM Delay Discounting

Measuring how large language models choose between smaller immediate rewards and larger delayed ones — and whether that discounting is exponential or hyperbolic.

## Overview

This project runs intertemporal choice trials on LLMs across six beneficiary conditions (self, future self, another AI, stranger, charity, and a no-framing control) and six delays (1 day to 5 years). It estimates the discount rate *k* per condition and tests whether hyperbolic or exponential discounting better describes the choices.

See [`preregistration.md`](preregistration.md) for the full hypotheses, design, and analysis plan.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then add your GEMINI_API_KEY
```

## Running

```bash
# Test the pipeline with no API calls
python3 src/run_experiment.py --model mock --limit 50

# Real run (from project root)
python3 src/run_experiment.py --model gemini_flash --limit 50

# Analyze results
python3 src/analysis.py
```

## Tests

```bash
python3 -m pytest tests/ -v
```

## Project structure

```
config/           experiment parameters (delays, amounts, beneficiaries, models)
src/
  prompts.py      trial and prompt generation
  models.py       Gemini / Ollama / mock backends
  run_experiment.py  CLI runner with resume support
  parse.py        A/B response parser
  analysis.py     indifference points, curve fitting, bootstrap CIs
data/raw/         raw JSONL output (gitignored)
data/processed/   parsed CSVs
results/          tables and figures
tests/            pytest suite
preregistration.md  hypotheses and analysis plan (written before data collection)
```
