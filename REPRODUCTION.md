# Counting Manifolds Reproduction

This checkout is prepared for the open-model reproduction of Anthropic's
"When Models Manipulate Manifolds: The Geometry of a Counting Task".

## Environment

Conda prefix:

```bash
conda activate /NAS/chennc/NashChennc/.tmp/conda-envs/counting-manifolds
```

The environment was installed from this repository's `poetry.lock` with
`POETRY_VIRTUALENVS_CREATE=false`.

## GPU Checks

Run these before launching a full reproduction:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

## Runs

Pythia-70M validates the full hidden-state, probe, and PCA pipeline:

```bash
HF_HOME=/NAS/chennc/NashChennc/.tmp/hf-cache CUDA_VISIBLE_DEVICES=0 python -m counting_manifolds.main --config_path configs/repro-pythia-70m-gpu.yml
```

GPT-2 additionally exercises the supported GPT-2 SAE path:

```bash
HF_HOME=/NAS/chennc/NashChennc/.tmp/hf-cache CUDA_VISIBLE_DEVICES=0 python -m counting_manifolds.main --config_path configs/repro-gpt2-gpu.yml
```

Outputs are written under:

```text
/NAS/chennc/NashChennc/results/counting-manifolds/<model>/<dataset>/
```

Expected key artifacts include `lm_metrics.csv`, `metrics.csv`,
`mean_hiddens.npy`, `explained_variance_ratio.npy`, and
`mean_hiddens_pca_slice_20_to_150_d6.npy`. The GPT-2 run should also emit SAE
artifacts such as `sae_mean_acts.npy` and `top_sae_features_projected.npy`.

## Reproducibility Notes

The original paper's Claude 3.5 Haiku analyses cannot be exactly reproduced
from public model weights. These configs reproduce the public open-model
counting-manifold pipeline in `corl-team/counting_manifolds`.
