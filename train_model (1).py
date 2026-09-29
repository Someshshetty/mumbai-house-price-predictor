"""
train_model.py
Trains three models to predict Mumbai house prices (in Rs lakh):
  1. Linear Regression            (baseline)
  2. Gradient Boosting Regressor  (main regression model)
  3. Deep Neural Network (MLP)    (deep learning model, 3 hidden layers)
Run:  python train_model.py
Outputs go to models/ (bundle.joblib, metrics.csv, importance.csv) and figures/ (PNG charts).
"""
import os
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = "data/mumbai_house_data.csv"
SEED = 42

TARGET = "price_lakh"
CAT_COLS = ["locality", "region", "status", "furnishing"]
NUM_COLS = ["bhk", "area_sqft", "floor", "total_floors", "age_years", "parking",
            "amenities_score", "dist_station_km", "sea_facing"]
FEATURES = CAT_COLS + NUM_COLS
# NOTE (leakage check): 'price_per_sqft' is deliberately NOT a feature - it is derived from the target.


def load_data(path=DATA_PATH):
    if not os.path.exists(path):
        import generate_data
        generate_data.build_dataset(path)   # creates the CSV automatically if it is missing
    df = pd.read_csv(path)
    df = df.drop_duplicates()
    df = df[(df[TARGET] > 0) & (df["area_sqft"] > 0)]
    # remove extreme price-per-sqft outliers (data errors) using the 0.5%-99.5% band
    psf = df[TARGET] * 1e5 / df["area_sqft"]
    lo, hi = psf.quantile([0.005, 0.995])
    df = df[(psf >= lo) & (psf <= hi)].reset_index(drop=True)
    return df


def make_preprocessor():
    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer([("num", num, NUM_COLS), ("cat", cat, CAT_COLS)])


def make_models():
    """All models predict log(price); predictions are converted back with exp()."""
    return {
        "Linear Regression (Baseline)": Pipeline([
            ("prep", make_preprocessor()), ("model", LinearRegression())]),
        "Gradient Boosting": Pipeline([
            ("prep", make_preprocessor()),
            ("model", HistGradientBoostingRegressor(
                max_iter=400, learning_rate=0.06, max_depth=6, l2_regularization=1.0,
                early_stopping=True, validation_fraction=0.1, random_state=SEED))]),
        "Deep Neural Network (MLP)": Pipeline([
            ("prep", make_preprocessor()),
            ("model", MLPRegressor(
                hidden_layer_sizes=(128, 64, 32), activation="relu", solver="adam",
                learning_rate_init=1e-3, alpha=1e-4, batch_size=64, max_iter=300,
                early_stopping=True, validation_fraction=0.1, n_iter_no_change=15,
                random_state=SEED))]),
    }


def metrics(y_true, y_pred):
    return {
        "MAE (Rs lakh)": mean_absolute_error(y_true, y_pred),
        "RMSE (Rs lakh)": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
        "MAPE (%)": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
    }


def predict_lakh(pipeline, X):
    return np.exp(pipeline.predict(X))


def make_figures(df, results, y_test, preds, mlp_pipe, importance):
    os.makedirs("figures", exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    # 1 price distribution
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    ax[0].hist(df[TARGET], bins=50, color="#4361ee"); ax[0].set_title("Price distribution (Rs lakh)")
    ax[1].hist(np.log(df[TARGET]), bins=50, color="#7209b7"); ax[1].set_title("log(Price) distribution")
    fig.tight_layout(); fig.savefig("figures/fig1_price_distribution.png", dpi=160); plt.close(fig)

    # 2 avg price per sq ft by locality
    psf = (df[TARGET] * 1e5 / df["area_sqft"]).groupby(df["locality"]).mean().sort_values()
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(psf.index, psf.values / 1000, color="#4361ee")
    ax.set_xlabel("Average price per sq ft (Rs '000)"); ax.set_title("Average price per sq ft by locality")
    fig.tight_layout(); fig.savefig("figures/fig2_locality_psf.png", dpi=160); plt.close(fig)

    # 3 area vs price
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.scatter(df["area_sqft"], df[TARGET], s=5, alpha=0.35, color="#3a0ca3")
    ax.set_xlabel("Carpet/built-up area (sq ft)"); ax.set_ylabel("Price (Rs lakh)")
    ax.set_title("Area vs price")
    fig.tight_layout(); fig.savefig("figures/fig3_area_vs_price.png", dpi=160); plt.close(fig)

    # 4 correlation heatmap (numeric)
    corr = df[NUM_COLS + [TARGET]].corr()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=60, ha="right")
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns)
    fig.colorbar(im, fraction=0.046); ax.set_title("Correlation matrix")
    fig.tight_layout(); fig.savefig("figures/fig4_correlation.png", dpi=160); plt.close(fig)

    # 5 feature importance
    imp = importance.sort_values("importance")
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.barh(imp["feature"], imp["importance"], color="#7209b7")
    ax.set_xlabel("Permutation importance (drop in R2)"); ax.set_title("Gradient Boosting - feature importance")
    fig.tight_layout(); fig.savefig("figures/fig5_feature_importance.png", dpi=160); plt.close(fig)

    # 6 MLP loss curve
    m = mlp_pipe.named_steps["model"]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(m.loss_curve_, label="Training loss", color="#4361ee")
    ax2 = ax.twinx(); ax2.plot(m.validation_scores_, color="#f72585", label="Validation R2")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Training loss (squared error / 2)"); ax2.set_ylabel("Validation R2")
    ax.set_title("Neural network training curve")
    fig.tight_layout(); fig.savefig("figures/fig6_mlp_training.png", dpi=160); plt.close(fig)

    # 7 actual vs predicted (all models)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8), sharex=True, sharey=True)
    lim = float(np.percentile(y_test, 99))
    for a, (name, p) in zip(ax, preds.items()):
        a.scatter(y_test, p, s=6, alpha=0.4, color="#4361ee"); a.plot([0, lim], [0, lim], "r--", lw=1)
        a.set_xlim(0, lim); a.set_ylim(0, lim); a.set_title(name, fontsize=9)
        a.set_xlabel("Actual (Rs lakh)")
    ax[0].set_ylabel("Predicted (Rs lakh)")
    fig.tight_layout(); fig.savefig("figures/fig7_actual_vs_predicted.png", dpi=160); plt.close(fig)

    # 8 metric comparison
    fig, ax = plt.subplots(figsize=(7, 4))
    names = list(results.index); x = np.arange(len(names)); w = 0.38
    ax.bar(x - w / 2, results["MAE (Rs lakh)"], w, label="MAE", color="#4361ee")
    ax.bar(x + w / 2, results["RMSE (Rs lakh)"], w, label="RMSE", color="#7209b7")
    ax.set_xticks(x); ax.set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=8)
    ax.set_ylabel("Error (Rs lakh)"); ax.set_title("Model error comparison (held-out test set)"); ax.legend()
    fig.tight_layout(); fig.savefig("figures/fig8_model_comparison.png", dpi=160); plt.close(fig)


def main():
    os.makedirs("models", exist_ok=True)
    df = load_data()
    print(f"Rows after cleaning: {len(df)}")

    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
    print(f"Train: {len(X_tr)}  Test: {len(X_te)}")

    models, rows, preds, cv_rows = make_models(), {}, {}, {}
    for name, pipe in models.items():
        pipe.fit(X_tr, np.log(y_tr))
        p = predict_lakh(pipe, X_te)
        preds[name] = p
        rows[name] = metrics(y_te.values, p)
        print(f"{name:32s}", {k: round(v, 3) for k, v in rows[name].items()})

    # 5-fold cross-validation (R2 on log-price) for the two regression models
    kf = KFold(5, shuffle=True, random_state=SEED)
    for name in list(models)[:2]:
        s = cross_val_score(make_models()[name], X_tr, np.log(y_tr), cv=kf, scoring="r2")
        cv_rows[name] = {"cv_r2_mean": float(s.mean()), "cv_r2_std": float(s.std())}
        print(f"CV R2 (log price) {name}: {s.mean():.4f} +/- {s.std():.4f}")

    results = pd.DataFrame(rows).T
    results.to_csv("models/metrics.csv")

    # permutation importance on the gradient boosting model (works on the raw input columns)
    gb = models["Gradient Boosting"]
    pi = permutation_importance(gb, X_te, np.log(y_te), n_repeats=5, random_state=SEED, scoring="r2")
    importance = pd.DataFrame({"feature": FEATURES, "importance": pi.importances_mean})
    importance.sort_values("importance", ascending=False).to_csv("models/importance.csv", index=False)

    # empirical prediction interval (10th-90th percentile of actual/predicted ratio) from the test set
    ratio = y_te.values / preds["Gradient Boosting"]
    interval = {"low": float(np.quantile(ratio, 0.10)), "high": float(np.quantile(ratio, 0.90))}

    # options for the Streamlit form
    loc_region = df.groupby("locality")["region"].first().to_dict()
    loc_stats = (df.assign(psf=df[TARGET] * 1e5 / df["area_sqft"])
                   .groupby("locality")["psf"].mean().round(0).to_dict())
    bundle = {"models": models, "results": results, "importance": importance, "interval": interval,
              "loc_region": loc_region, "loc_psf": loc_stats, "cv": cv_rows,
              "n_train": len(X_tr), "n_test": len(X_te), "n_total": len(df)}
    joblib.dump(bundle, "models/bundle.joblib", compress=3)

    make_figures(df, results, y_te.values, preds, models["Deep Neural Network (MLP)"], importance)
    with open("models/summary.json", "w") as f:
        json.dump({"results": rows, "cv": cv_rows, "interval": interval,
                   "n_total": len(df), "n_train": len(X_tr), "n_test": len(X_te),
                   "psf_mean": float((df[TARGET] * 1e5 / df["area_sqft"]).mean()),
                   "price_stats": df[TARGET].describe().to_dict()}, f, indent=2)
    print("Saved models/bundle.joblib and figures/")


if __name__ == "__main__":
    main()
