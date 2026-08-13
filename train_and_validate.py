"""Train and validate intent classification models for DriftCatch.
Usage: run with the workspace Python (preferably the project's venv).
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


def load_data(path="driftcatch.csv"):
    df = pd.read_csv(path)
    df = df.dropna(subset=["prompt", "label"])  # required columns
    return df


def build_preprocessor(text_col="prompt", numeric_features=None):
    if numeric_features is None:
        numeric_features = [
            "roleplay_indicator",
            "system_prompt_ref",
            "jailbreak_keyword_score",
            "word_count",
            "char_count",
        ]
    preprocessor = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(max_features=2000, ngram_range=(1, 2)), text_col),
            ("num", StandardScaler(), numeric_features),
        ],
        remainder="drop",
    )
    return preprocessor


def train_and_evaluate(df):
    text_feature = "prompt"
    numeric_features = [
        "roleplay_indicator",
        "system_prompt_ref",
        "jailbreak_keyword_score",
        "word_count",
        "char_count",
    ]

    X = df[[text_feature] + numeric_features]
    y = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(text_feature, numeric_features)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=7, random_state=42, n_jobs=-1),
    }

    results = {}

    for name, model in models.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
        print(f"Training {name}...")
        pipeline.fit(X_train, y_train)

        # Cross-validated score on training data (5-fold)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")
        print(f"  CV train accuracy (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        preds = pipeline.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"  Test accuracy: {acc:.4f}")
        print("  Classification report:\n", classification_report(y_test, preds))

        results[name] = {
            "pipeline": pipeline,
            "cv_mean": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "test_acc": float(acc),
            "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        }

    # Select best model by test accuracy
    best_name = max(results.keys(), key=lambda k: results[k]["test_acc"])
    best = results[best_name]
    print(f"Best model: {best_name} with test accuracy={best['test_acc']:.4f}")

    # Save best pipeline
    out_path = os.path.join(os.getcwd(), "firewall_model.pkl")
    joblib.dump(best["pipeline"], out_path)
    print(f"Saved best model to {out_path}")

    # Build and save metadata for semantic analysis (TF-IDF centroids per class)
    try:
        # Extract fitted vectorizer from pipeline preprocessor
        vect = best["pipeline"].named_steps["preprocessor"].named_transformers_["text"]
        # Transform training text to TF-IDF vectors
        X_text_train = vect.transform(X_train[text_feature])
        # Compute centroids (mean vector) for each class
        class_centroids = {}
        for cls in np.unique(y_train):
            mask = (y_train == cls).values
            if mask.sum() > 0:
                centroid = X_text_train[mask].mean(axis=0)
                # convert to dense array
                class_centroids[int(cls)] = np.asarray(centroid).ravel()

        meta = {
            "class_centroids": class_centroids,
            "text_feature": text_feature,
            "numeric_features": numeric_features,
            "vectorizer_shape": X_text_train.shape,
        }
        meta_path = os.path.join(os.getcwd(), "model_meta.pkl")
        joblib.dump(meta, meta_path)
        print(f"Saved model metadata to {meta_path}")
    except Exception as e:
        print("Warning: could not compute/save model metadata:", e)

    return results, best_name


def main():
    df = load_data()
    results, best_name = train_and_evaluate(df)


if __name__ == "__main__":
    main()
