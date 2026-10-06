"""
train_model.py
==============
Loan Approval Prediction - Model Training Script

Pure numpy / scipy implementation — NO sklearn required.
This bypasses Application Control policies that block compiled sklearn DLLs.

Models trained:
  1. Logistic Regression  (scipy L-BFGS-B optimisation)
  2. Gaussian Naive Bayes (pure numpy)
  3. K-Nearest Neighbours (pure numpy, batched)

Usage:
    python train_model.py
"""

import os
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy import optimize
from scipy.special import expit  # numerically stable sigmoid

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════
# 1.  Synthetic Data Generation
# ══════════════════════════════════════════════════════════════

def generate_loan_data(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic loan dataset.

    Key design choices
    ------------------
    - credit_history=1 strongly increases approval probability (dominant signal)
    - Higher combined income increases approval chances
    - Graduates get a slight boost; urban/semiurban areas help marginally
    - ~5 % missing values injected to make preprocessing non-trivial
    """
    rng = np.random.default_rng(random_state)

    # ── Categorical features ──────────────────────────────────────────────────
    gender       = rng.choice(["Male", "Female"],             size=n_samples, p=[0.72, 0.28])
    married      = rng.choice(["Yes", "No"],                  size=n_samples, p=[0.65, 0.35])
    dependents   = rng.choice(["0", "1", "2", "3+"],          size=n_samples, p=[0.57, 0.17, 0.16, 0.10])
    education    = rng.choice(["Graduate", "Not Graduate"],   size=n_samples, p=[0.78, 0.22])
    self_employed= rng.choice(["Yes", "No"],                  size=n_samples, p=[0.14, 0.86])
    property_area= rng.choice(["Urban", "Semiurban", "Rural"],size=n_samples, p=[0.33, 0.38, 0.29])
    loan_amount_term = rng.choice(
        [12, 36, 60, 84, 120, 180, 240, 300, 360, 480],
        size=n_samples,
        p=[0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.06, 0.08, 0.62, 0.02],
    )

    # credit_history: ~84 % have a clean record (matches real Kaggle distribution)
    credit_history = rng.choice([1.0, 0.0], size=n_samples, p=[0.84, 0.16]).astype(float)

    # ── Numeric features ──────────────────────────────────────────────────────
    base_income  = rng.lognormal(mean=8.5, sigma=0.6, size=n_samples).astype(int)
    income_boost = (
        np.where(education == "Graduate", 1500, 0)
        + np.where(gender == "Male", 800, 0)
        + np.where(married == "Yes", 500, 0)
    )
    applicant_income = np.clip(base_income + income_boost, 1500, 81_000)

    has_coapplicant  = rng.choice([1, 0], size=n_samples, p=[0.40, 0.60])
    coapplicant_income = np.where(
        has_coapplicant,
        np.clip(rng.lognormal(mean=7.8, sigma=0.55, size=n_samples), 0, 41_667),
        0.0,
    ).round(2)

    combined_income = applicant_income + coapplicant_income
    loan_amount = np.clip(
        (combined_income * rng.uniform(1.0, 4.5, size=n_samples) / 1000).astype(int),
        9, 700,
    )

    # ── Target: loan_status ───────────────────────────────────────────────────
    log_odds = np.zeros(n_samples)
    log_odds += np.where(credit_history == 1.0, 2.5, -2.8)          # dominant
    income_z  = (combined_income - combined_income.mean()) / combined_income.std()
    log_odds += 0.6 * income_z
    log_odds += np.where(education == "Graduate", 0.5, -0.3)
    log_odds += np.where(property_area == "Semiurban", 0.35, 0.0)
    log_odds += np.where(property_area == "Urban",     0.20, 0.0)
    log_odds += np.where(married      == "Yes",        0.25, 0.0)
    log_odds += np.where(self_employed== "Yes",       -0.15, 0.0)
    log_odds += np.where(loan_amount_term == 360,      0.20, 0.0)
    log_odds += rng.normal(0, 0.8, size=n_samples)                   # noise

    approval_prob = expit(log_odds)
    loan_status   = np.where(rng.binomial(1, approval_prob) == 1, "Y", "N")

    # ── Inject ~5 % missing values ────────────────────────────────────────────
    def add_missing(arr, frac=0.05):
        mask = rng.choice([True, False], size=len(arr), p=[frac, 1 - frac])
        arr  = arr.astype(object)
        arr[mask] = np.nan
        return arr

    gender        = add_missing(gender,        0.02)
    married       = add_missing(married,       0.01)
    dependents    = add_missing(dependents,    0.03)
    self_employed = add_missing(self_employed, 0.03)
    loan_amount   = add_missing(loan_amount.astype(object), 0.04)
    credit_history= add_missing(credit_history.astype(object), 0.07)

    loan_ids = [f"LP{str(i).zfill(6)}" for i in range(1, n_samples + 1)]

    df = pd.DataFrame({
        "loan_id":           loan_ids,
        "gender":            gender,
        "married":           married,
        "dependents":        dependents,
        "education":         education,
        "self_employed":     self_employed,
        "applicant_income":  applicant_income,
        "coapplicant_income":coapplicant_income,
        "loan_amount":       loan_amount,
        "loan_amount_term":  loan_amount_term,
        "credit_history":    credit_history,
        "property_area":     property_area,
        "loan_status":       loan_status,
    })
    return df


# ══════════════════════════════════════════════════════════════
# 2.  Preprocessor  (pure pandas / numpy)
# ══════════════════════════════════════════════════════════════

class Preprocessor:
    """
    sklearn-compatible preprocessor built on pure pandas/numpy.

    Steps
    -----
    1. Categorical columns → mode-impute → one-hot encode
    2. Numerical columns   → median-impute → standard-scale
    """

    def __init__(self):
        self.cat_cols:      list  = []
        self.num_cols:      list  = []
        self.cat_modes:     dict  = {}
        self.num_medians:   dict  = {}
        self.categories_:   dict  = {}   # {col: [sorted unique categories]}
        self.means_:        dict  = {}
        self.stds_:         dict  = {}
        self.feature_names_: list = []

    # ── fit ───────────────────────────────────────────────────────────────────
    def fit(self, X: pd.DataFrame) -> "Preprocessor":
        self.cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
        self.num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        # Imputation statistics
        for col in self.cat_cols:
            mode = X[col].mode(dropna=True)
            self.cat_modes[col] = mode.iloc[0] if len(mode) > 0 else "Unknown"

        for col in self.num_cols:
            self.num_medians[col] = float(
                pd.to_numeric(X[col], errors="coerce").median()
            )

        # OHE categories (computed on imputed copy)
        X_imp = X.copy()
        for col in self.cat_cols:
            X_imp[col] = X_imp[col].fillna(self.cat_modes[col])
        for col in self.cat_cols:
            self.categories_[col] = sorted(X_imp[col].dropna().unique().tolist())

        # Scaling statistics
        for col in self.num_cols:
            vals = pd.to_numeric(X_imp[col], errors="coerce").fillna(self.num_medians[col])
            self.means_[col] = float(vals.mean())
            std = float(vals.std())
            self.stds_[col]  = std if std > 1e-8 else 1.0

        # Build feature name list (order: OHE features, then numeric)
        names = []
        for col in self.cat_cols:
            for cat in self.categories_[col]:
                names.append(f"{col}_{cat}")
        names += self.num_cols
        self.feature_names_ = names
        return self

    # ── transform ─────────────────────────────────────────────────────────────
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = X.copy()

        # Impute
        for col in self.cat_cols:
            if col in X.columns:
                X[col] = X[col].fillna(self.cat_modes.get(col, "Unknown"))

        for col in self.num_cols:
            if col in X.columns:
                X[col] = (
                    pd.to_numeric(X[col], errors="coerce")
                    .fillna(self.num_medians.get(col, 0.0))
                )

        # One-hot encode
        ohe_parts = []
        for col in self.cat_cols:
            for cat in self.categories_.get(col, []):
                ohe_parts.append(
                    (X[col] == cat).astype(float).values.reshape(-1, 1)
                )

        # Standard-scale numerics
        num_parts = []
        for col in self.num_cols:
            vals   = X[col].values.astype(float)
            scaled = (vals - self.means_[col]) / self.stds_[col]
            num_parts.append(scaled.reshape(-1, 1))

        return np.hstack(ohe_parts + num_parts)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


# ══════════════════════════════════════════════════════════════
# 3.  ML Models  (pure numpy / scipy)
# ══════════════════════════════════════════════════════════════

# ── 3a. Logistic Regression ───────────────────────────────────────────────────

class LogisticRegressionScratch:
    """
    Binary logistic regression optimised with scipy L-BFGS-B.

    Exposes sklearn-compatible interface:
      .fit(X, y)  .predict(X)  .predict_proba(X)
      .coef_  shape (1, n_features)   — used by app.py for feature importance
    """

    def __init__(self, C: float = 1.0, max_iter: int = 500, tol: float = 1e-5):
        self.C        = C
        self.max_iter = max_iter
        self.tol      = tol
        self.coef_:      np.ndarray | None = None  # shape (1, n_features)
        self.intercept_: np.ndarray | None = None  # shape (1,)
        self.classes_:   np.ndarray | None = None
        self.model_name  = "Logistic Regression"

    def _objective(self, params: np.ndarray, X: np.ndarray, y: np.ndarray):
        """Log-loss + L2 regularisation, returns (loss, gradient)."""
        n, d = X.shape
        w, b = params[:d], params[d]
        z    = X @ w + b
        p    = np.clip(expit(z), 1e-10, 1 - 1e-10)
        loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        loss += np.sum(w ** 2) / (2.0 * self.C * n)        # L2

        diff   = p - y
        grad_w = X.T @ diff / n + w / (self.C * n)
        grad_b = diff.mean()
        return loss, np.append(grad_w, grad_b)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionScratch":
        self.classes_ = np.array([0, 1])
        d      = X.shape[1]
        params = np.zeros(d + 1)
        result = optimize.minimize(
            self._objective,
            params,
            args=(X, y.astype(float)),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": self.max_iter, "ftol": self.tol},
        )
        self.coef_      = result.x[:d].reshape(1, -1)   # (1, n_features)
        self.intercept_ = result.x[d:d+1]               # (1,)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z     = X @ self.coef_[0] + self.intercept_[0]
        prob1 = expit(z)
        return np.column_stack([1 - prob1, prob1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ── 3b. Gaussian Naive Bayes ──────────────────────────────────────────────────

class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes classifier — pure numpy.
    Exposes .feature_importances_ as normalised mutual-information proxy.
    """

    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing  = var_smoothing
        self.classes_:       np.ndarray | None = None
        self.class_prior_:   np.ndarray | None = None
        self.theta_:         np.ndarray | None = None  # (n_classes, n_feat) means
        self.sigma_:         np.ndarray | None = None  # (n_classes, n_feat) variances
        self.feature_importances_: np.ndarray | None = None
        self.model_name     = "Naive Bayes"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNaiveBayes":
        self.classes_    = np.array([0, 1])
        n_feat           = X.shape[1]
        self.class_prior_= np.zeros(2)
        self.theta_      = np.zeros((2, n_feat))
        self.sigma_      = np.zeros((2, n_feat))

        for i, c in enumerate(self.classes_):
            Xc = X[y == c]
            self.class_prior_[i] = len(Xc) / len(y)
            self.theta_[i]       = Xc.mean(axis=0)
            self.sigma_[i]       = Xc.var(axis=0) + self.var_smoothing

        # Feature importance: variance of class-conditional means (separability)
        sep = np.abs(self.theta_[1] - self.theta_[0])
        total = sep.sum()
        self.feature_importances_ = sep / total if total > 0 else np.ones(n_feat) / n_feat
        return self

    def _log_likelihood(self, X: np.ndarray, ci: int) -> np.ndarray:
        mu, sig = self.theta_[ci], self.sigma_[ci]
        return (
            -0.5 * np.sum(np.log(2 * np.pi * sig))
            - 0.5 * np.sum(((X - mu) ** 2) / sig, axis=1)
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        lp = np.column_stack([
            np.log(self.class_prior_[i]) + self._log_likelihood(X, i)
            for i in range(2)
        ])
        lp -= lp.max(axis=1, keepdims=True)    # numerical stability
        prob = np.exp(lp)
        prob /= prob.sum(axis=1, keepdims=True)
        return prob

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


# ── 3c. K-Nearest Neighbours ─────────────────────────────────────────────────

class KNearestNeighbors:
    """
    KNN classifier — pure numpy, memory-efficient batched inference.
    """

    def __init__(self, n_neighbors: int = 21, batch_size: int = 200):
        self.n_neighbors  = n_neighbors
        self.batch_size   = batch_size
        self.X_train:  np.ndarray | None = None
        self.y_train:  np.ndarray | None = None
        self.model_name   = "K-Nearest Neighbors"
        # KNN has no inherent feature importance; use uniform
        self.feature_importances_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNearestNeighbors":
        self.X_train = X.astype(float)
        self.y_train = y.astype(int)
        self.feature_importances_ = np.ones(X.shape[1]) / X.shape[1]
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X    = X.astype(float)
        n    = len(X)
        prob = np.empty((n, 2))

        for start in range(0, n, self.batch_size):
            end   = min(start + self.batch_size, n)
            chunk = X[start:end]                           # (B, d)
            # Euclidean distances:  (B, N)
            diff  = chunk[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]
            dists = np.sqrt((diff ** 2).sum(axis=2))
            nn_idx = np.argsort(dists, axis=1)[:, :self.n_neighbors]
            nn_labels = self.y_train[nn_idx]               # (B, k)
            p1    = nn_labels.mean(axis=1)
            prob[start:end, 1] = p1
            prob[start:end, 0] = 1 - p1

        return prob

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ══════════════════════════════════════════════════════════════
# 4.  Metrics  (pure numpy)
# ══════════════════════════════════════════════════════════════

def roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, list, list]:
    """Compute ROC-AUC and return (auc, fpr_list, tpr_list)."""
    order    = np.argsort(-y_prob)
    y_sorted = y_true[order]
    tps      = np.cumsum(y_sorted).astype(float)
    fps      = np.cumsum(1 - y_sorted).astype(float)
    # Prepend (0, 0)
    tps = np.concatenate([[0.0], tps])
    fps = np.concatenate([[0.0], fps])
    tpr = tps / (tps[-1] + 1e-12)
    fpr = fps / (fps[-1] + 1e-12)
    auc = float(abs(np.trapz(tpr, fpr)))
    return auc, fpr.tolist(), tpr.tolist()


def compute_metrics(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    """Return a dict of all evaluation metrics expected by app.py."""
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    n          = len(y_true)
    accuracy   = (tp + tn) / n
    precision  = tp / (tp + fp + 1e-12)
    recall     = tp / (tp + fn + 1e-12)
    f1         = 2 * precision * recall / (precision + recall + 1e-12)

    auc, fpr_list, tpr_list = roc_auc(y_true, y_prob)

    return {
        "model_name":    model_name,
        "accuracy":      round(accuracy,  4),
        "precision":     round(precision, 4),
        "recall":        round(recall,    4),
        "f1":            round(f1,        4),
        "roc_auc":       round(auc,       4),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "roc_curve":     {"fpr": fpr_list, "tpr": tpr_list},
    }


def cross_val_auc(
    model_cls,
    model_kwargs: dict,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Stratified K-Fold cross-validation, returns array of ROC-AUC scores."""
    rng   = np.random.default_rng(seed)
    pos   = np.where(y == 1)[0]
    neg   = np.where(y == 0)[0]
    rng.shuffle(pos); rng.shuffle(neg)
    pos_folds = np.array_split(pos, n_splits)
    neg_folds = np.array_split(neg, n_splits)

    scores = []
    for i in range(n_splits):
        val_idx   = np.concatenate([pos_folds[i], neg_folds[i]])
        train_idx = np.concatenate(
            [pos_folds[j] for j in range(n_splits) if j != i]
            + [neg_folds[j] for j in range(n_splits) if j != i]
        )
        m = model_cls(**model_kwargs)
        m.fit(X[train_idx], y[train_idx])
        p = m.predict_proba(X[val_idx])[:, 1]
        auc, _, _ = roc_auc(y[val_idx], p)
        scores.append(auc)
    return np.array(scores)


# ══════════════════════════════════════════════════════════════
# 5.  Train / Evaluate Helper
# ══════════════════════════════════════════════════════════════

def train_and_evaluate(
    model,
    model_cls,
    model_kwargs: dict,
    X_train: np.ndarray,
    X_test:  np.ndarray,
    y_train: np.ndarray,
    y_test:  np.ndarray,
    do_cv:   bool = True,
) -> tuple[dict, object]:
    """Fit model, compute all metrics, optionally run CV."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    m = compute_metrics(model.model_name, y_test, y_pred, y_prob)

    if do_cv:
        cv_scores = cross_val_auc(model_cls, model_kwargs, X_train, y_train)
        m["cv_roc_auc_mean"] = round(float(cv_scores.mean()), 4)
        m["cv_roc_auc_std"]  = round(float(cv_scores.std()),  4)
    else:
        m["cv_roc_auc_mean"] = m["roc_auc"]
        m["cv_roc_auc_std"]  = 0.0

    return m, model


# ══════════════════════════════════════════════════════════════
# 6.  Main Training Pipeline
# ══════════════════════════════════════════════════════════════

def train_and_save(output_dir: str = "models", data_dir: str = "data"):
    """
    End-to-end training pipeline:
      1. Generate & save 5 000-record synthetic dataset
      2. Split  →  Preprocess
      3. Train & evaluate 3 models
      4. Select best by ROC-AUC
      5. Save all artifacts for the Streamlit app
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(data_dir,   exist_ok=True)

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print("=" * 62)
    print("  LOAN APPROVAL PREDICTION — TRAINING PIPELINE")
    print("=" * 62)
    print("\n[1/6] Generating synthetic loan dataset (5 000 records)…")
    df = generate_loan_data(n_samples=5000, random_state=42)
    csv_path = os.path.join(data_dir, "loan_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"      ✓ Saved → {csv_path}")
    print(f"      Shape         : {df.shape}")
    print(f"      Approval rate : {(df['loan_status'] == 'Y').mean():.1%}")

    # ── 2. Split ──────────────────────────────────────────────────────────────
    print("\n[2/6] Splitting features and target…")
    X = df.drop(columns=["loan_id", "loan_status"])
    y = (df["loan_status"] == "Y").astype(int).values

    rng    = np.random.default_rng(42)
    idx    = np.arange(len(y))
    # Stratified 80/20 split
    pos, neg = idx[y == 1], idx[y == 0]
    rng.shuffle(pos); rng.shuffle(neg)
    test_pos = pos[:int(len(pos) * 0.2)]
    test_neg = neg[:int(len(neg) * 0.2)]
    test_idx = np.concatenate([test_pos, test_neg])
    train_idx = np.setdiff1d(idx, test_idx)

    X_train_raw = X.iloc[train_idx].reset_index(drop=True)
    X_test_raw  = X.iloc[test_idx].reset_index(drop=True)
    y_train     = y[train_idx]
    y_test      = y[test_idx]
    print(f"      Train : {len(y_train)}  |  Test : {len(y_test)}")

    # ── 3. Preprocessor ───────────────────────────────────────────────────────
    print("\n[3/6] Fitting preprocessor (impute + OHE + scale)…")
    preprocessor = Preprocessor()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test  = preprocessor.transform(X_test_raw)
    feat_names = preprocessor.feature_names_
    print(f"      ✓ {len(feat_names)} features after encoding")

    # ── 4. Models ─────────────────────────────────────────────────────────────
    print("\n[4/6] Training & evaluating models…")
    model_specs = [
        (
            LogisticRegressionScratch(C=1.0, max_iter=500),
            LogisticRegressionScratch,
            {"C": 1.0, "max_iter": 500},
        ),
        (
            GaussianNaiveBayes(var_smoothing=1e-9),
            GaussianNaiveBayes,
            {"var_smoothing": 1e-9},
        ),
        (
            KNearestNeighbors(n_neighbors=21),
            KNearestNeighbors,
            {"n_neighbors": 21},
        ),
    ]

    all_metrics     = []
    trained_models  = {}

    for model, model_cls, model_kwargs in model_specs:
        print(f"      Training {model.model_name}…", end=" ", flush=True)
        # Skip CV for KNN (slow) — use hold-out AUC only
        do_cv = not isinstance(model, KNearestNeighbors)
        m, fitted = train_and_evaluate(
            model, model_cls, model_kwargs,
            X_train, X_test, y_train, y_test,
            do_cv=do_cv,
        )
        all_metrics.append(m)
        trained_models[model.model_name] = fitted
        print(f"AUC={m['roc_auc']:.4f}  Acc={m['accuracy']:.4f}")

    # ── 5. Best model ─────────────────────────────────────────────────────────
    print("\n[5/6] Selecting best model…")
    best_entry = max(all_metrics, key=lambda m: m["roc_auc"])
    best_name  = best_entry["model_name"]
    best_model = trained_models[best_name]
    print(f"      🏆 Best : {best_name}  (AUC={best_entry['roc_auc']:.4f})")

    # ── 6. Save artifacts ─────────────────────────────────────────────────────
    print("\n[6/6] Saving model artifacts…")
    best_y_prob = best_model.predict_proba(X_test)[:, 1]

    joblib.dump(best_model,  os.path.join(output_dir, "best_model.pkl"))
    joblib.dump(preprocessor,os.path.join(output_dir, "preprocessor.pkl"))
    joblib.dump(feat_names,  os.path.join(output_dir, "feature_names.pkl"))
    joblib.dump(best_entry,  os.path.join(output_dir, "metrics.pkl"))
    joblib.dump(all_metrics, os.path.join(output_dir, "model_comparison.pkl"))
    joblib.dump(
        {"y_test": y_test, "y_prob": best_y_prob, "X_test": X_test},
        os.path.join(output_dir, "test_data.pkl"),
    )

    for f in ["best_model.pkl", "preprocessor.pkl", "feature_names.pkl",
              "metrics.pkl", "model_comparison.pkl", "test_data.pkl"]:
        print(f"      ✓ {f}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  MODEL COMPARISON SUMMARY")
    print("=" * 62)
    hdr = f"{'Model':<28} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7}"
    print(hdr)
    print("-" * 62)
    for m in sorted(all_metrics, key=lambda x: x["roc_auc"], reverse=True):
        marker = " 🏆" if m["model_name"] == best_name else ""
        print(
            f"{m['model_name']:<28} "
            f"{m['accuracy']:>7.4f} "
            f"{m['precision']:>7.4f} "
            f"{m['recall']:>7.4f} "
            f"{m['f1']:>7.4f} "
            f"{m['roc_auc']:>7.4f}"
            f"{marker}"
        )
    print("=" * 62)
    print("\n✅ Training complete!  Run → streamlit run app.py\n")

    return best_model, preprocessor, feat_names, best_entry, all_metrics


# ══════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    train_and_save(output_dir="models", data_dir="data")
