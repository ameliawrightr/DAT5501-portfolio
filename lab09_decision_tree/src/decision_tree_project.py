import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    precision_score, 
    recall_score,
    f1_score,
    confusion_matrix, 
    classification_report,
)

artifacts_dir = Path(__file__).resolve().parent.parent / "artifacts"
artifacts_dir.mkdir(parents=True, exist_ok=True)

#1. LOAD DATASET
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "winequalitywhite.csv"

print("Loading data from:", DATA_PATH)

df = pd.read_csv(DATA_PATH, sep=';')

#display first few rows
print("First few rows:")
print(df.head(),"\n")
print("Dataframe cols and types:")
print(df.info(),"\n")


#2. CLASSIFICATION TARGET
#orginal quality scores range from 3 to 9
#    good_quality = 1 if quality >=7 (good/excellent)
#    bad_quality = 0 if quality <=6 (average/bad)

df['good_quality'] = (df['quality'] >= 7).astype(int)

print("Class distribution (0 = bad, 1 = good):")
print(df['good_quality'].value_counts(normalize=True),"\n")


#3. DEFINE FEATURES (X) AND TARGET (y)

feature_cols = [
    col for col in df.columns
    if col not in ['quality', 'good_quality']   
]

X = df[feature_cols]
y = df['good_quality']

print(f"Number of features: {X.shape[1]}")
print("Feature columns:", feature_cols, "\n")


#4. SPLIT DATA INTO TRAINING AND TEST SETS
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y #keep class ratios similar in train/test
)

print(f"Train size: {X_train.shape[0]} samples")
print(f"Test size: {X_test.shape[0]} samples\n")

#5. TRAIN DECISION TREE CLASSIFIER
clf = DecisionTreeClassifier(
    criterion='gini',
    max_depth=5,
    random_state=42
)

clf.fit(X_train, y_train)

#5a. FEATURE IMPORTANCE

importances = clf.feature_importances_ #array - same order as X columns
feature_importance_df = X.columns #or feature_cols

#comb into (name, importance) pairs and sort by importance
indices = np.argsort(importances)[::-1]
print("Feature importances:")
for i in indices:
    print(f"{feature_importance_df[i]:25s}: {importances[i]:.3f}")

#5b. plot as bar chart
plt.figure()
plt.barh(
    [feature_importance_df[i] for i in indices],
    [importances[i] for i in indices]
)
plt.xlabel("Feature Importance")
plt.title("White Wine Feature Importances from Decision Tree Classifier")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(artifacts_dir / 'PLOT0_feature_importances.png', dpi=300)
plt.show()
plt.close()

#6. MAKE PREDICTIONS

y_pred = clf.predict(X_test)

#7. EVALUATE MODEL PERFORMANCE
precision = precision_score(y_test, y_pred, average="binary")
recall = recall_score(y_test, y_pred, average="binary")

print("EVALUATION ON TEST SET")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}\n")

print("Confusion matrix [rows=true, cols=pred]:")
print(confusion_matrix(y_test, y_pred),"\n")

print("Classification report:")
print(classification_report(y_test, y_pred, target_names=['bad_quality', 'good_quality']))


#8. DEPTH IMPORTANCE
depths = list(range(1, 16))

precisions = []
recalls = []
f1_scores = []

print("Effect of max_depth on model performance:")
for depth in depths:
    clf_tmp = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )
    clf_tmp.fit(X_train, y_train)
    y_pred_tmp = clf_tmp.predict(X_test)

    precisions.append(precision_score(y_test, y_pred_tmp, average="binary"))
    recalls.append(recall_score(y_test, y_pred_tmp, average="binary"))
    f1_scores.append(f1_score(y_test, y_pred_tmp, average="binary"))    

    print(f"max_depth={depth}: precision={precisions[-1]:.3f}, recall={recalls[-1]:.3f}")

#8b. PLOTTING DEPTH VS METRICS
fig = plt.figure(figsize=(10, 6))
plt.plot(depths, precisions, label='Precision', marker='o')
plt.plot(depths, recalls, label='Recall', marker='o')
plt.plot(depths, f1_scores, label='F1 Score', marker='o')

plt.xlabel("Tree Depth")
plt.ylabel("Score")
plt.title("Precision, Recall, F1 Score vs. Tree Depth")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

fig.savefig(artifacts_dir / 'PLOT1_precision_recall_f1_vs_tree_depth.png', dpi=300)
plt.close(fig)