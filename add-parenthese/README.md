# Add Parentheses Experiment

## Goal

Train a character-level language model to learn how to add parentheses around a word.

## Task Description

Given a word, the model should output the same word wrapped in parentheses.

### Example Data

hello (hello)
Youssif (Youssif)
cat (cat)
machine (machine)

## Motivation

This is a simple sequence transformation task that can be learned purely at the character level. It serves as a baseline experiment to verify that the model can learn deterministic string transformations.

## Training on Mac

NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
python -u ../../comp560-nanoGPT/train.py config/basic.py \
--dataset=basic \
--device=cpu \
--compile=False \
--eval_iters=20 \
--log_interval=1 \
--block_size=64 \
--batch_size=12 \
--n_layer=3 \
--n_head=3 \
--n_embd=120 \
--max_iters=200 \
--lr_decay_iters=200 \
--dropout=0.0

## Training on Mac that follows basic.py

NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python -u ../../comp560-nanoGPT/train.py config/basic.py \
--dataset=basic \
--device=cpu \
--compile=False

## Sampling on Mac

NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py \
python -u ../../comp560-nanoGPT/sample.py config/basic.py \
--dataset=basic \
--device=cpu \
--num_samples=1 \
--max_new_tokens=100 \
--seed=2345
