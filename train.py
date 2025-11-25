#!/usr/bin/env python3
"""
Train a regression model for house prices.
Supports:
 - California Housing (sklearn) [easy]
 - Kaggle CSV (user-provided): expects CSV with target column 'SalePrice' or 'target' (you can rename).
Saves:
 - model.joblib  (best fitted sklearn Pipeline)
 - preprocessor.joblib (if separate)
 - metrics.json
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.datasets import fetch_california_housing

def load_kaggle(path: Path):
    df = pd.read_csv(path)
    # Try find target column
    if 'SalePrice' in df.columns:
        y = df['SalePrice'].copy()
        X = df.drop(columns=['SalePrice'])
    elif 'target' in df.columns:
        y = df['target'].copy()
        X = df.drop(columns=['target'])
    else:
        raise ValueError("CSV must contain 'SalePrice' or 'target' column as target.")
    return X, y

def load_california():
    data = fetch_california_housing(as_frame=True)
    X = data.frame.drop(columns=['MedHouseVal'], errors='ignore')  # note dataset target is MedHouseVal
    y = data.frame['MedHouseVal']
    return X, y

def build_preprocessor(X: pd.DataFrame):
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=['number']).columns.tolist()

    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    if categorical_cols:
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])
        preprocessor = ColumnTransformer([
            ('num', numeric_pipeline, numeric_cols),
            ('cat', cat_pipeline, categorical_cols)
        ])
    else:
        preprocessor = ColumnTransformer([
            ('num', numeric_pipeline, numeric_cols)
        ])

    return preprocessor, numeric_cols, categorical_cols

def main(args):
    if args.kaggle_csv:
        X, y = load_kaggle(Path(args.kaggle_csv))
    else:
        X, y = load_california()

    # small clean for Kaggle: drop columns with >80% missing
    X = X.loc[:, X.isnull().mean() < 0.8]

    preprocessor, num_cols, cat_cols = build_preprocessor(X)

    # We'll search over Ridge and Lasso; also keep a plain LinearRegression baseline.
    # Create pipeline with a placeholder estimator (we'll GridSearch over estimator type using 'estimator' param)
    from sklearn.base import clone
    # We'll define three separate pipelines and compare with simple CV
    pipelines = {
        'linear': Pipeline([('pre', preprocessor), ('model', LinearRegression())]),
        'ridge': Pipeline([('pre', preprocessor), ('model', Ridge())]),
        'lasso': Pipeline([('pre', preprocessor), ('model', Lasso(max_iter=10000))])
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42)

    results = {}
    best_pipeline = None
    best_score = -np.inf

    for name, pipe in pipelines.items():
        if name == 'linear':
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            results[name] = {'mse': float(mse), 'r2': float(r2), 'mae': float(mae)}
            print(f"{name} -> r2={r2:.4f}, mse={mse:.4f}, mae={mae:.4f}")
            if r2 > best_score:
                best_score = r2
                best_pipeline = pipe
        else:
            # grid search alpha
            alphas = np.logspace(-3, 3, 25)
            param_grid = {'model__alpha': alphas}
            g = GridSearchCV(pipe, param_grid, cv=5, scoring='r2', n_jobs=args.n_jobs, verbose=0)
            g.fit(X_train, y_train)
            best = g.best_estimator_
            y_pred = best.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            results[name] = {
                'best_alpha': float(g.best_params_['model__alpha']),
                'mse': float(mse),
                'r2': float(r2),
                'mae': float(mae)
            }
            print(f"{name} best alpha={g.best_params_['model__alpha']:.4g} -> r2={r2:.4f}, mse={mse:.4f}, mae={mae:.4f}")
            if r2 > best_score:
                best_score = r2
                best_pipeline = best

    # Save best pipeline
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / 'model.joblib'
    joblib.dump(best_pipeline, model_path)
    print(f"Saved best model to {model_path}")

    # Save metadata and columns for Streamlit
    meta = {
        'numeric_columns': num_cols,
        'categorical_columns': cat_cols,
        'results': results,
        'dataset_rows': int(X.shape[0]),
        'target_mean': float(y.mean()),
        'target_std': float(y.std()),
    }
    with open(out_dir / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    # Save full results
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("Training complete. Metrics:")
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kaggle-csv', type=str, default=None, help='Path to Kaggle CSV with target SalePrice or target column. If omitted, uses sklearn California dataset.')
    parser.add_argument('--out-dir', type=str, default='models', help='Output directory to save model and meta')
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--n-jobs', type=int, default=1)
    args = parser.parse_args()
    main(args)
