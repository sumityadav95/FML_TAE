from flask import Flask, render_template, request, redirect, url_for, flash
import os
import uuid
import traceback
import numpy as np
import pandas as pd

from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-this-secret-key")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
UPLOAD_FOLDER = os.path.join(app.root_path, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {"csv"}

MODELS = {
    "Logistic Regression": (
        LogisticRegression(max_iter=2000),
        {"model__C": [0.1, 1, 10], "model__solver": ["lbfgs"]}
    ),
    "Decision Tree": (
        DecisionTreeClassifier(random_state=42),
        {"model__max_depth": [None, 5, 10, 20], "model__min_samples_split": [2, 5, 10]}
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=42, n_jobs=-1),
        {"model__n_estimators": [50, 100, 150], "model__max_depth": [None, 5, 10],
         "model__min_samples_split": [2, 5]}
    ),
    "K-Nearest Neighbors": (
        KNeighborsClassifier(),
        {"model__n_neighbors": [3, 5, 7, 9], "model__weights": ["uniform", "distance"]}
    ),
    "Support Vector Machine": (
        SVC(),
        {"model__C": [0.1, 1, 10], "model__kernel": ["linear", "rbf"]}
    )
}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def make_pipeline(X, estimator):
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    transformers = []
    if numeric_cols:
        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("numeric", numeric_pipe, numeric_cols))
    if categorical_cols:
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])
        transformers.append(("categorical", categorical_pipe, categorical_cols))

    preprocess = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline([("preprocess", preprocess), ("model", estimator)])

def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred) * 100, 2),
        "precision": round(precision_score(y_true, y_pred, average="weighted", zero_division=0) * 100, 2),
        "recall": round(recall_score(y_true, y_pred, average="weighted", zero_division=0) * 100, 2),
        "f1": round(f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100, 2)
    }

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", models=list(MODELS.keys()))

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("dataset")
    if not file or file.filename == "":
        flash("Please choose a CSV file.")
        return redirect(url_for("index"))
    if not allowed_file(file.filename):
        flash("Only CSV files are supported.")
        return redirect(url_for("index"))

    token = uuid.uuid4().hex
    path = os.path.join(UPLOAD_FOLDER, token + ".csv")
    file.save(path)
    try:
        df = pd.read_csv(path)
        if df.empty or len(df.columns) < 2:
            raise ValueError("The CSV must contain at least two columns and one data row.")
        if len(df) < 10:
            raise ValueError("Please upload at least 10 rows for a meaningful train/test split.")
        preview = df.head(8).replace({np.nan: ""}).to_dict(orient="records")
        return render_template(
            "configure.html", token=token, columns=df.columns.tolist(),
            preview=preview, row_count=len(df), col_count=len(df.columns),
            models=list(MODELS.keys())
        )
    except Exception as exc:
        if os.path.exists(path):
            os.remove(path)
        flash(f"Could not read the dataset: {exc}")
        return redirect(url_for("index"))

@app.route("/optimize", methods=["POST"])
def optimize():
    token = request.form.get("token", "")
    target = request.form.get("target")
    model_name = request.form.get("model")
    method = request.form.get("method", "grid")

    if not token.isalnum() or model_name not in MODELS or method not in {"grid", "random"}:
        flash("Invalid form selection. Please upload your dataset again.")
        return redirect(url_for("index"))

    path = os.path.join(UPLOAD_FOLDER, token + ".csv")
    if not os.path.isfile(path):
        flash("Dataset upload expired. Please upload it again.")
        return redirect(url_for("index"))

    try:
        df = pd.read_csv(path)
        if target not in df.columns:
            raise ValueError("Please select a valid target column.")
        df = df.dropna(subset=[target])
        if len(df) < 10:
            raise ValueError("Not enough rows remain after removing missing target values.")

        y = df[target]
        X = df.drop(columns=[target])
        if X.shape[1] == 0:
            raise ValueError("Your dataset needs at least one feature column.")
        if y.nunique() < 2:
            raise ValueError("The target column must contain at least two classes.")
        if y.nunique() > 50 and y.nunique() / len(y) > 0.5:
            raise ValueError("The selected target looks like an ID or continuous value. Choose a classification target.")

        # Stratify where possible; otherwise use a regular split.
        stratify = y if y.value_counts().min() >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify
        )

        estimator, param_grid = MODELS[model_name]
        baseline_pipeline = make_pipeline(X_train, estimator)
        baseline_pipeline.fit(X_train, y_train)
        baseline_pred = baseline_pipeline.predict(X_test)
        baseline_metrics = calculate_metrics(y_test, baseline_pred)

        search_pipeline = make_pipeline(X_train, estimator)
        if method == "grid":
            search = GridSearchCV(search_pipeline, param_grid, cv=3, scoring="accuracy",
                                  n_jobs=-1, error_score="raise")
            method_label = "Grid Search"
        else:
            search = RandomizedSearchCV(search_pipeline, param_grid, n_iter=min(8, 12),
                                        cv=3, scoring="accuracy", random_state=42,
                                        n_jobs=-1, error_score="raise")
            method_label = "Random Search"

        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        optimized_pred = best_model.predict(X_test)
        optimized_metrics = calculate_metrics(y_test, optimized_pred)

        labels = sorted(pd.Series(y_test).astype(str).unique().tolist())
        cm = confusion_matrix(y_test.astype(str), pd.Series(optimized_pred).astype(str), labels=labels).tolist()
        best_params = {k.replace("model__", ""): str(v) for k, v in search.best_params_.items()}

        # Remove uploaded data after successful processing.
        os.remove(path)
        return render_template(
            "results.html", model_name=model_name, method_label=method_label,
            baseline=baseline_metrics, optimized=optimized_metrics,
            best_params=best_params, labels=labels, confusion=cm,
            cv_score=round(search.best_score_ * 100, 2),
            improvement=round(optimized_metrics["accuracy"] - baseline_metrics["accuracy"], 2)
        )
    except Exception as exc:
        app.logger.exception("Optimization failed")
        if os.path.exists(path):
            os.remove(path)
        flash(f"Optimization could not be completed: {exc}")
        return redirect(url_for("index"))

@app.errorhandler(413)
def too_large(_error):
    flash("File is too large. Maximum upload size is 16 MB.")
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
