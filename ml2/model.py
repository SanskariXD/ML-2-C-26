"""Early-stopped tri-GBDT and held-out pair / candidate-context calibration."""
from __future__ import annotations
import importlib.metadata
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .data import write_json
from .features import CARE_NAMES, FEATURE_NAMES, SCHEMA_HASH, context_features
from .text import NORMALIZER_VERSION


def sigmoid(z):
    return 1. / (1. + np.exp(-np.clip(z, -60, 60)))


def fit_logistic(x, y, sample_weight) -> dict:
    if len(set(y.tolist())) < 2:
        raise ValueError("Calibration needs both positive and negative retrieved pairs; increase query sample")
    estimator = make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=1000, random_state=26))
    estimator.fit(x, y, logisticregression__sample_weight=sample_weight)
    scaler, lr = estimator.steps[0][1], estimator.steps[1][1]
    # Store numeric parameters as JSON rather than untrusted pickle.
    return {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
            "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0])}


def logistic_predict(x, params) -> np.ndarray:
    z = (x - np.array(params["mean"])) / np.array(params["scale"])
    return sigmoid(z @ np.array(params["coef"]) + params["intercept"])


class Ensemble:
    def __init__(self, models=None, weights=None):
        self.models = models or []
        self.weights = np.asarray(weights if weights is not None else [.4, .35, .25], dtype=np.float64)
        self.calibrators = {}
        self.thresholds = {}
        self.config = {}
        self.manifest = {}

    def fit(self, x, y, w, xe, ye, we, rounds=250, threads=2):
        import xgboost as xgb
        import lightgbm as lgb
        from catboost import CatBoostClassifier, Pool
        if len(set(y.tolist())) < 2 or len(set(ye.tolist())) < 2:
            raise ValueError("Fit and early-stop partitions need both classes; increase the sample")
        m1 = xgb.XGBClassifier(n_estimators=rounds, max_depth=5, learning_rate=.05, subsample=.85,
                              colsample_bytree=.9, tree_method="hist", n_jobs=threads, random_state=26,
                              eval_metric="logloss", early_stopping_rounds=20)
        m1.fit(x, y, sample_weight=w, eval_set=[(xe, ye)], sample_weight_eval_set=[we], verbose=False)
        m2 = lgb.LGBMClassifier(n_estimators=rounds, num_leaves=31, learning_rate=.05, max_depth=-1,
                               min_child_samples=10, verbosity=-1, n_jobs=threads, random_state=26,
                               deterministic=True, force_col_wise=True)
        m2.fit(x, y, sample_weight=w, eval_set=[(xe, ye)], eval_sample_weight=[we],
               callbacks=[lgb.early_stopping(20, verbose=False)])
        m3 = CatBoostClassifier(iterations=rounds, depth=5, learning_rate=.05, loss_function="Logloss",
                                thread_count=threads, random_seed=26, verbose=False, allow_writing_files=False)
        m3.fit(Pool(x, y, weight=w), eval_set=Pool(xe, ye, weight=we), early_stopping_rounds=20, use_best_model=True)
        self.models = [m1, m2, m3]
        return self

    def probabilities(self, x) -> np.ndarray:
        if len(x) == 0:
            return np.empty((0, 3), dtype=np.float32)
        result = []
        for m in self.models:
            # A loaded LightGBM Booster has predict rather than predict_proba.
            if hasattr(m, "booster_"):
                result.append(m.booster_.predict(x, num_iteration=m.best_iteration_))
            elif hasattr(m, "predict_proba"):
                result.append(m.predict_proba(x)[:, 1])
            else:
                result.append(m.predict(x))
        return np.column_stack(result)

    def design(self, x, probs, mode):
        if mode == "care":
            return context_features(x, probs, self.weights)
        if mode != "pair":
            raise ValueError(f"Unknown policy {mode}")
        p = np.clip(probs @ self.weights, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def score_query(self, x, mode="pair"):
        if not len(x):
            return np.empty(0, dtype=np.float64)
        return logistic_predict(self.design(x, self.probabilities(x), mode), self.calibrators[mode])

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.models[0].save_model(directory / "xgb.ubj")
        self.models[1].booster_.save_model(str(directory / "lgb.txt"))
        self.models[2].save_model(str(directory / "cat.cbm"))
        self.manifest.update({"schema_hash": SCHEMA_HASH, "feature_names": list(FEATURE_NAMES), "care_names": list(CARE_NAMES),
                         "normalizer": NORMALIZER_VERSION, "weights": self.weights.tolist(),
                         "calibrators": self.calibrators, "thresholds": self.thresholds, "config": self.config,
                         "versions": {n: importlib.metadata.version(n) for n in ["numpy", "scikit-learn", "xgboost", "lightgbm", "catboost", "rapidfuzz"]}})
        write_json(directory / "manifest.json", self.manifest)

    @classmethod
    def load(cls, directory):
        import xgboost as xgb
        import lightgbm as lgb
        from catboost import CatBoostClassifier
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest.get("schema_hash") != SCHEMA_HASH or manifest.get("feature_names") != list(FEATURE_NAMES):
            raise ValueError("Incompatible model feature schema; source 55/72-feature artifacts are not loadable here")
        if manifest.get("normalizer") != NORMALIZER_VERSION or manifest.get("care_names") != list(CARE_NAMES):
            raise ValueError("Preprocessing/context schema mismatch")
        for package, expected in manifest["versions"].items():
            if importlib.metadata.version(package) != expected:
                raise ValueError(f"Artifact dependency mismatch: install {package}=={expected}")
        m1, m3 = xgb.XGBClassifier(), CatBoostClassifier()
        m1.load_model(directory / "xgb.ubj")
        m3.load_model(str(directory / "cat.cbm"))
        m2 = lgb.Booster(model_file=str(directory / "lgb.txt"))
        model = cls([m1, m2, m3], manifest["weights"])
        model.calibrators, model.thresholds = manifest["calibrators"], manifest["thresholds"]
        model.config, model.manifest = manifest["config"], manifest
        return model
