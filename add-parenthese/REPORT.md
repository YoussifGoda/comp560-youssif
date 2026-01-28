# NanoGPT Basic Model Experiment Report

**Objective:**  
To experiment with a miniature character-level GPT model that can complete simple strings consisting of words and their parenthetical forms (e.g., `apple (apple)`) and to iteratively improve its performance on CPU without long training times.

---

## Dataset

- **Initial dataset:** Small set of building blocks (words + parentheses), e.g., `hello (hello)`, `cat (cat)`, `dog (dog)`, etc.
- **Size:** ~1,000,000 characters (~1MB).
- **Preparation:** `prepare.py` randomly repeated these building blocks to generate the training data. Both training and validation sets are identical to simplify overfitting testing.
- **Unique characters:** 23 (letters, spaces, parentheses, newline).

**Changes made:**

- Maintained the same building blocks but ensured enough repetitions for meaningful learning.
- Verified encoding and decoding worked correctly with `meta.pkl`.
- **Size:** ~500,000 characters (~0.5MB).

---

## Model & Config Tweaks

**Initial configuration:**

- 3 layers, 3 heads, 120 embedding dim (`n_embd`)
- Batch size 12, block size 64
- 200 iterations on CPU, learning rate 1e-3
- Dropout 0.0
- Output overfitting observed, but small dataset made training feasible.

**Tweaks for improvement:**

1. **Model size increased slightly:**
   - `n_layer` → 4
   - `n_head` → 4
   - `n_embd` → 128
   - Resulted in a larger but still lightweight model (~0.79M parameters).

2. **Training iterations increased:**
   - `max_iters` → 500
   - `lr_decay_iters` → 500
   - Allowed the model to further reduce loss on the tiny dataset.

3. **Sampling improvements:**
   - `temperature` = 0.7 (moderate randomness)
   - `top_k` = 20 (top token filtering)
   - Produced more coherent completions while keeping variety.

4. **CPU compatibility:**
   - Ensured `compile=False` and `device='cpu'` to avoid GPU/CUDA errors.
   - Batch size and block size kept small for memory efficiency.

---

## Training Observations

- **Initial loss:** ~3.17 → **Final loss:** ~0.33
- Model overfits the dataset quickly due to its small size.
- Even with CPU-only training, the model produces coherent outputs after ~200–300 iterations.

---

## Sampling Results

- Example completions after training:

```

computer (computer)
learning (learning)
hello (hello)
youssif (youssif)
banana (banana)
```

- Model reliably completes building blocks with parentheses correctly, occasionally truncating at block boundaries.
- Overall, outputs are more coherent and consistent compared to initial runs.

---

## Summary of Tweaks & Lessons Learned

1. **Increased model capacity slightly** to allow better learning without drastically increasing training time.
2. **Extended training iterations** for improved loss reduction.
3. **Tuned sampling parameters** (`temperature` and `top_k`) to improve output quality.
4. **Maintained CPU-only, small batch/sequence sizes** to avoid crashes and memory issues.
5. Dataset simplicity ensures overfitting is achievable, which is perfect for testing model behavior.

**Conclusion:**  
The experiment demonstrates that even a tiny GPT can learn to reproduce structured patterns like `word (word)` with minor tweaks. The model is now stable, produces reasonable completions, and can serve as a baseline for further experiments (e.g., adding parentheses logic, learning slightly larger vocabularies, or experimenting with temperature).

```

```
