<div align="center">

# 🎭 Bard-LSTM

**Shakespearean next-word prediction and text generation with stacked LSTMs, plus a PixelRNN study on image generation.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LaTeX](https://img.shields.io/badge/Report-LaTeX-008080?logo=latex&logoColor=white)](https://www.latex-project.org/)

*Assignment 2 · Deep Learning · Author: Husnain*

</div>

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Installation](#-installation)
- [Usage](#-usage)
- [Training Artifacts](#-training-artifacts)
- [Evaluation](#-evaluation)
- [Report](#-report)
- [Tech Stack](#-tech-stack)
- [Future Work](#-future-work)
- [Acknowledgements](#-acknowledgements)
- [Author](#-author)

---

## 📖 Overview

This repository contains the full submission for **Assignment 2**, which has two parts:

- **Question 1: PixelRNN.** An exploration of autoregressive image generation with recurrent networks. Figures and generated images are in the `Q1/` folder.
- **Question 2: Shakespeare Language Model.** A stacked LSTM trained on Shakespeare's plays to predict the next word in a sequence and generate new text in the Bard's style. It includes a complete training pipeline, evaluation tooling, and an interactive **Streamlit** web app.

Both parts are documented in a single combined LaTeX report (`main.tex`).

---

## ✨ Features

- 🧹 **Preprocessing pipeline:** text cleaning, tokenization, and n-gram sequence generation
- 🧠 **Stacked LSTM architecture** with dropout regularization
- ⏱ **Training callbacks:** early stopping, best-model checkpointing, and learning-rate scheduling
- 📉 **Evaluation tools:** training curves, validation loss, and **perplexity**
- 🖥 **Interactive Streamlit app** for live next-word prediction and text generation
- 🖼 **Report image generator** for example outputs and model comparisons
- 💾 **Reproducible runs:** tokenizer, configuration, history, and data splits are all saved to disk

---

## 📁 Project Structure

```
A2/
├─ app.py                      # Streamlit app (Q2)
├─ train.py                    # Training script (Q2)
├─ utils/
│  ├─ preprocessing.py         # Cleaning, tokenization, sequence generation
│  ├─ model_builder.py         # Stacked LSTM builder + callbacks
│  └─ evaluation.py            # Curves, perplexity, summaries
├─ image_genearte.py           # Generates report images (examples/comparison)
├─ data/
│  └─ shakespeare_plays.csv    # Dataset (Kaggle: kingburrito666/shakespeare-plays)
├─ models/                     # Saved artifacts after training
│  ├─ best_model.keras | final_model.keras
│  ├─ tokenizer.pickle | config.json | history.csv | run_config.json
│  └─ X_train.npy | y_train.npy | X_val.npy | y_val.npy
├─ reports/plots/              # Plots used by the LaTeX report
│  └─ training_curves.png | q2_examples.png | q2_compare.png | q2_streamlit.png
├─ main.tex                    # Combined LaTeX report (Q1 + Q2)
├─ requirements.txt            # Python dependencies
└─ Q1/                         # PixelRNN images and figures
```

---

## 📚 Dataset

The model is trained on the [Shakespeare Plays](https://www.kaggle.com/datasets/kingburrito666/shakespeare-plays) dataset from Kaggle, which contains the lines of Shakespeare's plays along with metadata such as play name, speaker, and act/scene/line numbers.

Download the CSV and place it at:

```
data/shakespeare_plays.csv
```

Only the spoken text (`PlayerLine`) is used for language modeling.

---

## 🔬 Methodology

1. **Cleaning:** lowercase the text, strip unwanted characters, and normalize whitespace.
2. **Tokenization:** fit a Keras `Tokenizer` on the corpus and build the vocabulary.
3. **Sequence generation:** create n-gram prefixes from each line, left-pad them to a fixed length, and split each into an input sequence and a next-word label.
4. **Train/validation split:** the arrays are saved to `models/` as `.npy` files for reproducibility.
5. **Model:** `Embedding → LSTM → Dropout → LSTM → Dropout → Dense (softmax)`.
6. **Training:** categorical cross-entropy loss with the Adam optimizer, using `EarlyStopping`, `ModelCheckpoint`, and `ReduceLROnPlateau`.
7. **Inference:** the model predicts a probability distribution over the vocabulary. The Streamlit app uses it for next-word suggestions and iterative text generation.

---

## ⚙️ Installation

**Prerequisites:** Python 3.10 or higher and `pip`.

```bash
# 1. Clone the repository
git clone https://github.com/husnain/A2.git
cd A2

# 2. (Recommended) Create a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Usage

### 1. Train the model

```bash
python train.py
```

This preprocesses the dataset, trains the stacked LSTM, and writes all artifacts to `models/`.

### 2. Launch the Streamlit app

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal (usually `http://localhost:8501`).

### 3. Generate report images

```bash
python image_genearte.py
```

Output images are saved to `reports/plots/`.

### 4. Build the LaTeX report

```bash
pdflatex main.tex
pdflatex main.tex
```

Run it twice to resolve references and the table of contents.

---

## 💾 Training Artifacts

| File                                   | Purpose                                      |
| -------------------------------------- | -------------------------------------------- |
| `best_model.keras`                     | Best checkpoint by validation loss           |
| `final_model.keras`                    | Model at the end of training                 |
| `tokenizer.pickle`                     | Fitted tokenizer used for inference          |
| `config.json`                          | Model and preprocessing configuration        |
| `run_config.json`                      | Hyperparameters of the specific training run |
| `history.csv`                          | Per-epoch loss and metric history            |
| `X_train.npy`, `y_train.npy`           | Training data                                |
| `X_val.npy`, `y_val.npy`               | Validation data                              |

---

## 📊 Evaluation

The model is evaluated with:

- **Training and validation curves** for loss and accuracy (`reports/plots/training_curves.png`)
- **Perplexity**, computed as `exp(validation loss)`, where lower is better
- **Qualitative examples** of generated text (`reports/plots/q2_examples.png`)
- **Model comparisons** across configurations (`reports/plots/q2_compare.png`)

Detailed results and analysis are in the report (`main.tex`).

---

## 📄 Report

`main.tex` is a combined LaTeX report covering:

- **Q1:** the PixelRNN study and its figures
- **Q2:** the Shakespeare language model, including the architecture, training setup, results, and Streamlit demo (`reports/plots/q2_streamlit.png`)

---

## 🛠 Tech Stack

| Category        | Technology                                  |
| --------------- | ------------------------------------------- |
| Language        | Python                                      |
| Deep Learning   | TensorFlow / Keras                          |
| Data Handling   | NumPy, Pandas                               |
| Visualization   | Matplotlib                                  |
| Web App         | Streamlit                                   |
| Documentation   | LaTeX                                       |

---

## 🚧 Future Work

- Add beam search and temperature or top-k sampling for more diverse generation
- Experiment with bidirectional LSTMs, GRUs, and Transformer-based models
- Move from word-level to subword or character-level tokenization
- Add pretrained embeddings such as GloVe
- Add unit tests for the preprocessing and evaluation utilities
- Deploy the Streamlit app publicly

---

## 🙏 Acknowledgements

- [Shakespeare Plays dataset](https://www.kaggle.com/datasets/kingburrito666/shakespeare-plays) by kingburrito666 on Kaggle
- The TensorFlow/Keras and Streamlit open-source communities

---

## 👤 Author

**Husnain**

Built as part of Assignment 2 for the Deep Learning course.

<div align="center">

⭐ If you found this project helpful, consider giving it a star!

</div>
