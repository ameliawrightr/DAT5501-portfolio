# Lab 09 – Decision Tree Mini-Project

This lab trains and evaluates a decision tree classifier on the UCI **White Wine Quality** dataset. The goal is to predict whether a wine is “good quality” and explore how model depth and feature importance affect performance.

---

## 1. Dataset

- Source: UCI Machine Learning Repository – *Wine Quality (White)*  
- File used: `data/winequalitywhite.csv`
- Features: 11 continuous physicochemical measurements (e.g. acidity, sulphates, alcohol).
- Target:
  - Original `quality` score (0–10) is converted to a binary label:
  - `good_quality = 1` if `quality ≥ 7`, else `0`.

---

## 2. Project Structure

```text
lab09_decision_tree/
  data/
    winequalitywhite.csv
  src/
    decision_tree_project.py
  README.md
  requirements.txt
````

---

## 3. Setup

From the repo root:

```bash
python -m venv sklearn-env
source sklearn-env/bin/activate

cd lab09_decision_tree
pip install -r requirements.txt
```

`requirements.txt` includes:

* `pandas`
* `scikit-learn`
* `matplotlib`
* `numpy`

---

## 4. Running the code

From the repo root (with the venv activated):

```bash
python lab09_decision_tree/src/decision_tree_project.py
```

The script will:

1. Load and inspect the dataset.
2. Create the `good_quality` binary target.
3. Split the data into train and test sets (stratified).
4. Train a `DecisionTreeClassifier`.
5. Print **precision**, **recall**, confusion matrix and a classification report.
6. Sweep over a range of `max_depth` values and plot precision, recall and F1 vs depth.
7. Plot feature importances as a horizontal bar chart.

Plots are shown interactively via `matplotlib`.

---

## 5. Model and Optimisation

* **Model:** `DecisionTreeClassifier` (Gini impurity, tunable `max_depth`).
* **Metrics:** Precision, recall, F1 on the held-out test set.
* **Depth optimisation:**
  We vary `max_depth` (e.g. 1–15) and observe:

  * Very shallow trees underfit (low scores).
  * Performance improves up to a moderate depth.
  * Deeper trees give little gain and can overfit (noisy or declining F1).

The “optimal” depth is chosen as the region where F1 is highest without excessive complexity.

---

## 6. Feature Importance

Using `clf.feature_importances_`:

* The script prints a sorted list of feature importances.
* A bar plot highlights which wine properties contribute most to predicting `good_quality`.

These importances can be used for simple feature selection (e.g. retraining with only the top k features and comparing metrics).