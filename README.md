# ✨ Predictive Maintenance: Turbofan Engine RUL Prediction

This repository contains a modular and extensible machine learning pipeline for predicting the **Remaining Useful Life (RUL)** of turbofan engines using the **NASA C-MAPSS** dataset. It supports multiple datasets (FD001-FD004), includes configuration-based control, and offers both interactive and CLI modes for usability.

---

## ⌛ Table of Contents

* [Overview](#overview)
* [Features](#features)
* [Dataset](#dataset)
* [Project Structure](#project-structure)
* [Setup & Installation](#setup--installation)
* [Configuration](#configuration)
* [Usage](#usage)

  * [Interactive Menu](#interactive-menu)
  * [Command-Line Interface (CLI)](#command-line-interface-cli)
* [Methodology](#methodology)
* [Outputs](#outputs)
* [Future Work](#future-work)
* [License](#license)

---

## 📈 Overview

Predictive maintenance aims to reduce downtime and maintenance costs in critical systems. This project focuses on predicting the RUL of aircraft engines using sensor data and machine learning models. A `RandomForestRegressor` model is trained on historical simulation data and used for evaluation on unseen data.

## ✨ Features

* **Multi-Dataset Support**: Compatible with FD001–FD004 subsets of C-MAPSS.
* **Centralized Configuration**: Modify settings in `config/config.py`.
* **Data Preprocessing**: Sensor pruning, scaling, and RUL labeling.
* **Feature Engineering**:

  * Rolling statistics (mean, std) over engine cycles.
  * Configurable rolling window size and RUL cap.
* **Modeling Pipeline**:

  * `StandardScaler` + `RandomForestRegressor` via `scikit-learn` pipeline.
  * Model persistence for reuse.
* **Evaluation Metrics**:

  * MSE, RMSE, R² score.
  * Visualization: Predicted vs Actual RUL.
* **Exploratory Data Analysis (EDA)**:

  * Dataset summary, correlation maps, distributions.
  * Outputs saved to `exploration_outputs/`.
* **User Interface**:

  * Interactive menu (no CLI knowledge required).
  * Command-line mode for advanced use.

## 📊 Dataset

Dataset: [NASA C-MAPSS](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#turbofan)

Each subset simulates different operational conditions and failure modes.

**Required Files per Subset (e.g., FD001):**

* `train_FD001.txt`
* `test_FD001.txt`
* `RUL_FD001.txt`

Ensure these files are placed in the `data/` directory.

## 📁 Project Structure

```
CMAPSS-PREDICTION/
├── config/
│   ├── __init__.py
│   └── config.py
├── data/                  # <- Place data files here (gitignored)
├── exploration_outputs/   # <- EDA plots and results (gitignored)
├── models/                # <- Trained models (gitignored)
├── src/
│   ├── __init__.py
│   ├── data_exploration.py
│   ├── data_processing.py
│   ├── evaluation.py
│   ├── feature_engineering.py
│   ├── model.py
│   └── visualization.py
├── main.py
├── README.md
└── requirements.txt
```

## ⚙️ Setup & Installation

```bash
# 1. Clone the repository
$ git clone <your-repository-url>
$ cd CMAPSS-PREDICTION

# 2. Create virtual environment (optional but recommended)
$ python -m venv venv
$ source venv/bin/activate        # On Windows use: venv\Scripts\activate

# 3. Install dependencies
$ pip install -r requirements.txt

# 4. Download and place C-MAPSS dataset into ./data/

# 5. Optional: Pre-create output folders
$ mkdir models exploration_outputs
```

## 💡 Configuration

Edit `config/config.py` to control:

* `WINDOW_SIZE`: Rolling window length.
* `RUL_CAP`: Cap for RUL value.
* `DATASETS`: Contains per-dataset file paths and sensor pruning info.
* `get_config(dataset_name)`: Fetch merged config for a dataset.

## ▶️ Usage

### Interactive Menu

Run without arguments:

```bash
python main.py
```

You will be prompted with options to:

* Train models
* Evaluate models
* Run EDA
* Exit

### Command-Line Interface (CLI)

```bash
python main.py <mode> <dataset_name1> [<dataset_name2> ...]
```

**Modes:**

* `train`: Train and save models.
* `evaluate`: Predict on test data and evaluate.
* `explore`: Run EDA.

**Example:**

```bash
python main.py train FD001 FD003
```

You can also run EDA standalone:

```bash
python -m src.data_exploration FD001
```

---

## 🧠 Methodology

### Configuration & Data Loading

* Loads dataset-specific settings via `get_config()`.
* Parses sensor data files using `pandas`, adds meaningful column names.

### Sensor Pruning & Feature Engineering

* Drops irrelevant sensors.
* Adds rolling mean/std features per engine unit.
* Labels training data with RUL (capped optionally).

### Modeling Pipeline

* Uses `ColumnTransformer` to scale selected columns.
* Trains `RandomForestRegressor` on preprocessed training data.
* Saves model to `models/<dataset_name>_model.pkl`.

### Evaluation

* Loads model and prepares test data (last cycle per engine).
* Evaluates with MSE, RMSE, R².
* Shows predicted vs actual RUL scatterplot.

---

## 📊 Outputs

* **Models**: `.pkl` files in `/models/`
* **Evaluation**: Console metrics + scatterplot
* **EDA**: PNG plots + summaries in `/exploration_outputs/<dataset>/`

---

## 🚀 Future Work

* Parameter tuning (`GridSearchCV`, `RandomizedSearchCV`)
* Deep learning support (LSTM, Transformers)
* Advanced error analysis
* Time-aware cross-validation
* Sensor auto-selection via feature importance
* Full time-series modeling (using complete test history)

---