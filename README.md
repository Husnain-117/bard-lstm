


## Project Structure

```
A2/
├─ app.py                      # Streamlit app (Q2)
├─ train.py                    # Training script (Q2)
├─ utils/
│  ├─ preprocessing.py         # Cleaning, tokenization, sequence generation (Q2)
│  ├─ model_builder.py         # Stacked LSTM builder + callbacks (Q2)
│  └─ evaluation.py            # Curves, perplexity, summaries (Q2)
├─ image_genearte.py           # Generate report images (Q2 examples/comparison)
├─ data/
│  └─ shakespeare_plays.csv    # Dataset (Kaggle: kingburrito666/shakespeare-plays)
├─ models/                     # Saved artifacts after training (Q2)
│  ├─ best_model.keras | final_model.keras
│  ├─ tokenizer.pickle | config.json | history.csv | run_config.json
│  ├─ X_train.npy | y_train.npy | X_val.npy | y_val.npy
├─ reports/plots/              # Plots used by LaTeX
│  ├─ training_curves.png | q2_examples.png | q2_compare.png | q2_streamlit.png
├─ main.tex                    # Combined LaTeX report (Q1 + Q2)
├─ requirements.txt            # Python deps
└─ Q1/                         # PixelRNN images and figures
```

---

## Environment Setup

- Python 3.10 recommended.
- Install dependencies:

```bash
pip install -r requirements.txt
```

If you see a TensorFlow oneDNN message about numerical differences, it’s informational. To disable oneDNN ops:

```bash
set TF_ENABLE_ONEDNN_OPTS=0  # Windows PowerShell: $env:TF_ENABLE_ONEDNN_OPTS=0
```

---

## Dataset (Q2)

Source: Kaggle — `kingburrito666/shakespeare-plays`.

1) Place your Kaggle API key at `%USERPROFILE%\.kaggle\kaggle.json`.
2) Download + unzip to `data/`:

```bash
kaggle datasets download -d kingburrito666/shakespeare-plays -p data
tar -xf data/*.zip -C data  # or Expand-Archive on Windows
```

Ensure the CSV path is `data/shakespeare_plays.csv` or pass the actual filename to `--csv`.

---

## Quick Start (Q2)

Smoke test on a subset (verifies pipeline):

```bash
python train.py --csv "data/shakespeare_plays.csv" --max_lines 5000 --epochs 3 --batch 128
```

Run the app:

```bash
streamlit run app.py
```

Open the browser UI, enter a prompt like `To be or not to`, and click Predict.

---

## Full Training (Q2)

Baseline configuration (recommended):

```bash
python train.py \
  --csv "data/shakespeare_plays.csv" \
  --epochs 50 --batch 128 --vocab 15000 --max_len 30 \
  --emb 256 --lstm 256 256 128
```

Artifacts saved to `models/`:

- `best_model.keras` (best by val accuracy), `final_model.keras`
- `tokenizer.pickle`, `config.json`, `history.csv`, `run_config.json`
- Numpy arrays for train/val splits

Curves saved to `reports/plots/training_curves.png` (and copied to `q2_training_curves.png` for LaTeX).

---

## Experiments (Q2)

Try deeper/wider/regularized models:

```bash
# Deep
python train.py --csv data/shakespeare_plays.csv --epochs 60 --batch 128 --emb 256 --lstm 512 512 256

# Wide
python train.py --csv data/shakespeare_plays.csv --epochs 50 --batch 128 --emb 300 --lstm 512 512

# Regularized
python train.py --csv data/shakespeare_plays.csv --epochs 60 --batch 128 --emb 256 --lstm 256 256
```

Generate figures for the report (examples and comparison):

```bash
python image_genearte.py
```

Outputs:

- `reports/plots/q2_examples.png` — top-5 predictions for multiple prompts
- `reports/plots/q2_compare.png` — val accuracy and perplexity across runs
- Add a Streamlit screenshot manually as `reports/plots/q2_streamlit.png`

---

## Streamlit App (Q2)

`app.py` loads `models/best_model.keras` or `final_model.keras` and the tokenizer/config.

Features:

- Temperature slider (0.5–1.5)
- Top-K selection (1–10)
- Top-5 predictions with probabilities
- Download training history

---

## Report Build

`main.tex` compiles both Q1 and Q2 sections.

```bash
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

Ensure the following figures exist before compiling:

- `reports/plots/training_curves.png`
- `reports/plots/q2_examples.png`
- `reports/plots/q2_compare.png`
- `reports/plots/q2_streamlit.png`

---

## Troubleshooting

- **FileNotFoundError: data/shakespeare_plays.csv**
  - Ensure the CSV exists or pass an absolute path via `--csv`.

- **Missing tokenizer/config** when generating figures
  - Run training once to create `models/tokenizer.pickle` and `models/config.json`.

- **TopK metric shape/type error**
  - We use `SparseTopKCategoricalAccuracy` to match integer labels. Ensure your install matches `requirements.txt`.

- **Slow training**
  - Use a GPU or reduce `--lstm` units and `--epochs` for testing.

---

## Notes (Q1)

- PixelRNN Image Completion figures live under `Q1/weights/samples/`.
- The LaTeX report includes methodology, results, and discussion for Q1.

---

## License

This repository is for academic coursework. Datasets are subject to their original licenses.
