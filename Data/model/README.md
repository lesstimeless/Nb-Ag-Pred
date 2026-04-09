# Model Directory

## Purpose
Pretraining and finetuning model information should be stored here after running:
- `mlm_pretrain.py`
- `binding_finetuning.py`

## Contents
This folder should contain:
- Train and validation losses for hyperparameter tuning across epochs

## Note
**Model Weights (.pt) are too large to be stored here** — run the model to generate the weights.