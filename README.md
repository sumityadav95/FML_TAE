# Hyperparameter Optimization for Classification Models

A working Flask application that accepts a CSV dataset, lets the user choose a target column and classification model, tunes hyperparameters with Grid Search or Random Search, and compares baseline and tuned test-set metrics.

## 1. Requirements
- Python 3.9 or newer
- pip

## 2. Install
Open a terminal in this project folder and run:

```bash
python -m venv venv
```

Activate it:

**Windows**
```bash
venv\Scripts\activate
```

**macOS / Linux**
```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 3. Run

```bash
python app.py
```

Open this address in your browser:

http://127.0.0.1:5000

## 4. Use
1. Upload a CSV file (up to 16 MB).
2. Select the column you want to predict.
3. Choose a classification model.
4. Choose Grid Search or Random Search.
5. Run optimization and review the results.

## Notes
- This application is designed for classification datasets with at least 10 rows and two target classes.
- Numeric missing values are median-imputed; categorical missing values are filled with the most frequent value. Categorical features are one-hot encoded.
- Data is split into 80% training and 20% testing. Hyperparameter search uses 3-fold cross-validation on the training set.
- Baseline and optimized models are evaluated on the same held-out test set.
- Scores are dataset-dependent; the application does not guarantee that tuning will improve test-set performance.
- Uploaded CSV files are removed after successful processing or an optimization error. For a production deployment, add authentication, stricter file validation, upload cleanup, CSRF protection, and a production WSGI server.
