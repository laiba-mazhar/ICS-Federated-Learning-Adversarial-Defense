# Detecting Adversarial Data Poisoning in Federated Learning Using Independent Class-wise Coherence Score (ICS)

Generative AI course project (Spring 2026, FAST-NUCES). Supervisor: Dr. Nouman Noor.

## Overview

Federated Learning (FL) lets many clients train a shared model without exchanging raw data,
but this leaves the global model exposed to malicious clients who poison their local data
(e.g. targeted label-flipping, backdoors). Existing coherence-based defenses judge client
trustworthiness only by *global* prediction agreement, which misses class-specific poisoning.

This project introduces the **Independent Class-wise Coherence Score (ICS)**, a defense that
scores each client's reliability *per class* using adversarially generated MI-FGSM probes. ICS
combines four coherence metrics — Output Consistency, Class Confidence, Inverse Entropy, and
Prediction Consistency — into a single weighted score used for selective aggregation.

Evaluated on MNIST, EMNIST, and CIFAR-10 under targeted label-flipping attacks, ICS reaches
75.02% accuracy with a 19.75% Attack Success Rate (ASR) on MNIST — a 5-7% accuracy gain and
30-40% ASR reduction over global-coherence baselines. Ablation studies confirm each ICS
component contributes to the result.

## Repo layout

- `src/ics_federated_experiment.py` — main federated training loop, ICS scoring, and MI-FGSM probe generation
- `src/run_ablation_grid.py` — runs the ablation grid (disabling one ICS component at a time) over the datasets
- `results/` — convergence plots, ICS heatmaps, per-dataset logs, and ablation comparisons for MNIST/EMNIST/CIFAR-10
- `report/` — IEEE-format LaTeX paper (`main.tex`) with figures and references

## Running

```bash
pip install -r requirements.txt
python src/ics_federated_experiment.py
python src/run_ablation_grid.py
```

Datasets (MNIST/EMNIST/CIFAR-10) are downloaded automatically via `torchvision` on first run
and are not included in this repo.
