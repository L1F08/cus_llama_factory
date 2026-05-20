"""Post-hoc calibration evaluation for cls_head / token-CE prediction files.

Path B of the optimization plan: fit a calibrator on val (or via K-fold on test)
so that the best-F1 threshold lands near 0.5, making default-0.5 deployment usable.

Reads the SAME prediction format as eval_qwen35_nat_logits.py:
    [{"id": "<video_path>", "logits": {"安全": float, "高风险": float}, ...}, ...]
GT JSONL:
    {"id": "...", "label": "高风险" | "安全" | 0 | 1}

Calibration target is z = logit_高风险 - logit_安全. Both inference scripts
(token-CE and cls_head) already produce this exact pair of raw logits, so this
script works on both unchanged.

Usage:
    python eval_calibration.py \\
        --pred /path/to/result.json \\
        --gt   /path/to/test.jsonl \\
        [--calib_pred /path/to/val_result.json --calib_gt /path/to/val.jsonl] \\
        [--kfold 5] [--target_thresh 0.5] \\
        [--save_calibrator /path/to/calibrator.json]

If --calib_pred is NOT provided, the script uses K-fold on the test set itself
(default K=5) so the reported numbers are NOT optimistically biased.
A final calibrator fitted on ALL test data is also saved for deployment.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Tuple

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import roc_auc_score, precision_recall_curve
    from sklearn.model_selection import StratifiedKFold
except ImportError:
    raise SystemExit("Need scikit-learn: pip install scikit-learn")

from scipy.optimize import minimize_scalar


# ====================== Data loading ======================
def load_pairs(pred_json_path: str, gt_jsonl_path: str) -> Tuple[np.ndarray, np.ndarray, list]:
    """Return (y_true [N], z [N], video_paths [N]) where z = logit_高风险 - logit_安全."""
    gt = {}
    with open(gt_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            raw = str(d["label"]).strip()
            if raw == "高风险":
                y = 1
            elif raw == "安全":
                y = 0
            else:
                y = int(raw)
            gt[str(d["id"])] = y

    with open(pred_json_path, "r", encoding="utf-8") as f:
        preds = json.load(f)

    ys, zs, paths = [], [], []
    skipped = 0
    for p in preds:
        vid = Path(p["id"]).stem
        if vid not in gt:
            skipped += 1
            continue
        logits = p.get("logits")
        if not logits or "安全" not in logits or "高风险" not in logits:
            skipped += 1
            continue
        try:
            z = float(logits["高风险"]) - float(logits["安全"])
        except (TypeError, ValueError):
            skipped += 1
            continue
        ys.append(gt[vid])
        zs.append(z)
        paths.append(p["id"])

    print(f"[load] {pred_json_path}")
    print(f"       N={len(ys)} (skipped {skipped}), pos={sum(ys)}, neg={len(ys)-sum(ys)}")
    return np.asarray(ys, dtype=np.int64), np.asarray(zs, dtype=np.float64), paths


# ====================== Calibrators ======================
EPS = 1e-12


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class Identity:
    """No calibration: P = sigmoid(z)."""
    name = "none"

    def fit(self, z, y):
        return self

    def predict_proba(self, z):
        return _sigmoid(z)

    def params(self):
        return {}


class TemperatureScaling:
    """P = sigmoid(z / T). Single positive scalar T fitted by NLL minimization.

    NOTE: T cannot shift the decision boundary z*=0, only sharpen/flatten the
    distribution. Included as a baseline to demonstrate why it's insufficient
    for under-confident cls_head (best thresh stays put).
    """
    name = "ts"

    def fit(self, z, y):
        def nll(logT):
            T = math.exp(logT)
            p = np.clip(_sigmoid(z / T), EPS, 1 - EPS)
            return -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
        res = minimize_scalar(nll, bounds=(-3.0, 3.0), method="bounded")
        self.T = math.exp(res.x)
        return self

    def predict_proba(self, z):
        return _sigmoid(z / self.T)

    def params(self):
        return {"T": self.T}


class Platt:
    """P = sigmoid(a * z + b). Standard Platt scaling (logistic regression on raw z).

    The bias b actually shifts the decision boundary, so this IS the method that
    can pull the best-F1 threshold back to 0.5.
    """
    name = "platt"

    def fit(self, z, y):
        self.lr = LogisticRegression(C=1e9, solver="lbfgs", max_iter=2000)
        self.lr.fit(z.reshape(-1, 1), y)
        self.a = float(self.lr.coef_[0, 0])
        self.b = float(self.lr.intercept_[0])
        return self

    def predict_proba(self, z):
        return _sigmoid(self.a * z + self.b)

    def params(self):
        return {"a": self.a, "b": self.b}


class Beta:
    """Beta calibration (Kull 2017): P = sigmoid(a*log(p) + b*log(1-p) + c)
    where p = sigmoid(z). Strictly more flexible than Platt; often better
    fits the asymmetric S-curve produced by focal loss.
    """
    name = "beta"

    def fit(self, z, y):
        p = np.clip(_sigmoid(z), EPS, 1 - EPS)
        X = np.column_stack([np.log(p), -np.log(1 - p)])
        self.lr = LogisticRegression(C=1e9, solver="lbfgs", max_iter=2000)
        self.lr.fit(X, y)
        self.a = float(self.lr.coef_[0, 0])
        self.b = float(self.lr.coef_[0, 1])
        self.c = float(self.lr.intercept_[0])
        return self

    def predict_proba(self, z):
        p = np.clip(_sigmoid(z), EPS, 1 - EPS)
        return _sigmoid(self.a * np.log(p) - self.b * np.log(1 - p) + self.c)

    def params(self):
        return {"a": self.a, "b": self.b, "c": self.c}


class Isotonic:
    """Non-parametric monotonic calibration. Robust but needs more data than parametric."""
    name = "isotonic"

    def fit(self, z, y):
        p = _sigmoid(z)
        self.ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.ir.fit(p, y)
        return self

    def predict_proba(self, z):
        return self.ir.transform(_sigmoid(z))

    def params(self):
        return {"type": "isotonic_nonparametric"}


CALIBRATORS = [Identity, TemperatureScaling, Platt, Beta, Isotonic]


# ====================== Metrics ======================
def expected_calibration_error(y_true: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(y_true)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if not mask.any():
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = p[mask].mean()
        ece += (mask.sum() / N) * abs(bin_acc - bin_conf)
    return float(ece)


def confusion_at(y_true: np.ndarray, p: np.ndarray, thresh: float):
    pred = (p >= thresh).astype(np.int64)
    TP = int(((pred == 1) & (y_true == 1)).sum())
    FP = int(((pred == 1) & (y_true == 0)).sum())
    TN = int(((pred == 0) & (y_true == 0)).sum())
    FN = int(((pred == 0) & (y_true == 1)).sum())
    f1 = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0.0
    return TP, FP, TN, FN, f1


def best_f1_sweep(y_true: np.ndarray, p: np.ndarray):
    """Exact best-F1 threshold via precision_recall_curve."""
    prec, rec, thr = precision_recall_curve(y_true, p)
    best_f1, best_t = -1.0, 0.5
    for i in range(len(thr)):
        if prec[i] + rec[i] > 0:
            f1 = 2 * prec[i] * rec[i] / (prec[i] + rec[i])
            if f1 > best_f1:
                best_f1, best_t = f1, float(thr[i])
    return best_f1, best_t


# ====================== Calibration eval driver ======================
def kfold_calibrated_probs(z: np.ndarray, y: np.ndarray, cls, k: int, seed: int) -> np.ndarray:
    """K-fold honest calibrated probabilities: fit on K-1 folds, predict on held-out."""
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    p_out = np.empty_like(z, dtype=np.float64)
    for tr_idx, te_idx in skf.split(z, y):
        calib = cls().fit(z[tr_idx], y[tr_idx])
        p_out[te_idx] = calib.predict_proba(z[te_idx])
    return p_out


def evaluate_one_calibrator(y, z, cls, kfold, seed, target_thresh):
    if cls is Identity:
        p = _sigmoid(z)
    else:
        p = kfold_calibrated_probs(z, y, cls, kfold, seed)
    p = np.clip(p, 0.0, 1.0)

    auc = roc_auc_score(y, p)
    best_f1, best_t = best_f1_sweep(y, p)
    TP, FP, TN, FN, f1_at = confusion_at(y, p, target_thresh)
    ece = expected_calibration_error(y, p)
    return {
        "name": cls.name,
        "auc": auc,
        "best_f1": best_f1,
        "best_thresh": best_t,
        "f1_at_target": f1_at,
        "TP_at": TP, "FP_at": FP, "TN_at": TN, "FN_at": FN,
        "ece": ece,
    }


def fit_final_calibrator(y, z, cls):
    """Fit calibrator on ALL data for deployment. Returns params dict."""
    if cls is Identity:
        return None
    calib = cls().fit(z, y)
    return {"method": cls.name, "params": calib.params()}


def print_report(rows, target_thresh, pred_json_path):
    print()
    print("=" * 110)
    print(f"📂 Prediction file: {pred_json_path}")
    print(f"🎯 Target deployment threshold: {target_thresh}")
    print("   (K-fold honest evaluation — calibrator never sees its own evaluation samples)")
    print("=" * 110)
    header = f"{'method':<10}|{'AUC':>8}|{'BestF1':>8}|{'BestThr':>8}|{'F1@target':>11}|{'FN@target':>10}|{'FP@target':>10}|{'ECE':>8}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['name']:<10}|"
            f"{r['auc']:>8.4f}|"
            f"{r['best_f1']:>8.4f}|"
            f"{r['best_thresh']:>8.4f}|"
            f"{r['f1_at_target']:>11.4f}|"
            f"{r['FN_at']:>10d}|"
            f"{r['FP_at']:>10d}|"
            f"{r['ece']:>8.4f}"
        )
    print("=" * 110)
    base = rows[0]  # identity
    print("\n📊 Improvement vs no calibration (Δ at target threshold):")
    for r in rows[1:]:
        d_f1 = r["f1_at_target"] - base["f1_at_target"]
        d_fn = r["FN_at"] - base["FN_at"]
        d_fp = r["FP_at"] - base["FP_at"]
        d_thr = r["best_thresh"] - base["best_thresh"]
        arrow = "✅" if d_f1 > 0 and abs(r["best_thresh"] - target_thresh) < abs(base["best_thresh"] - target_thresh) else "  "
        print(
            f"  {arrow} {r['name']:<8}  ΔF1@{target_thresh}={d_f1:+.4f}  ΔFN={d_fn:+d}  ΔFP={d_fp:+d}  "
            f"BestThr {base['best_thresh']:.3f}→{r['best_thresh']:.3f}"
        )
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="Prediction JSON (output of infer_faster_qwen35_*.py)")
    ap.add_argument("--gt", required=True, help="GT JSONL")
    ap.add_argument("--calib_pred", default=None,
                    help="Optional separate prediction file (e.g. on val set) to fit calibrator")
    ap.add_argument("--calib_gt", default=None, help="GT JSONL for --calib_pred (defaults to --gt)")
    ap.add_argument("--kfold", type=int, default=5,
                    help="K-fold count when --calib_pred is not given (default 5)")
    ap.add_argument("--target_thresh", type=float, default=0.5,
                    help="Deployment target threshold to report metrics at (default 0.5)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_calibrator", default=None,
                    help="Where to dump final-calibrator params (fitted on ALL test data)")
    args = ap.parse_args()

    y, z, _ = load_pairs(args.pred, args.gt)

    # If a separate calibration set is given, use HOLDOUT mode instead of K-fold.
    calib_yz = None
    if args.calib_pred is not None:
        calib_gt = args.calib_gt or args.gt
        cy, cz, _ = load_pairs(args.calib_pred, calib_gt)
        calib_yz = (cy, cz)
        print(f"[mode] HOLDOUT — fit on {args.calib_pred} (N={len(cy)}), eval on {args.pred} (N={len(y)})")
    else:
        print(f"[mode] K-FOLD (K={args.kfold}) on test set itself")

    rows = []
    for cls in CALIBRATORS:
        if calib_yz is None:
            row = evaluate_one_calibrator(y, z, cls, args.kfold, args.seed, args.target_thresh)
        else:
            cy, cz = calib_yz
            if cls is Identity:
                p = _sigmoid(z)
            else:
                calib = cls().fit(cz, cy)
                p = calib.predict_proba(z)
            p = np.clip(p, 0.0, 1.0)
            auc = roc_auc_score(y, p)
            best_f1, best_t = best_f1_sweep(y, p)
            TP, FP, TN, FN, f1_at = confusion_at(y, p, args.target_thresh)
            row = {
                "name": cls.name, "auc": auc, "best_f1": best_f1, "best_thresh": best_t,
                "f1_at_target": f1_at, "TP_at": TP, "FP_at": FP, "TN_at": TN, "FN_at": FN,
                "ece": expected_calibration_error(y, p),
            }
        rows.append(row)

    print_report(rows, args.target_thresh, args.pred)

    # Fit final deployable calibrator on ALL data (test for K-fold mode, or calib_pred if given)
    if args.save_calibrator:
        if calib_yz is None:
            fit_y, fit_z = y, z
        else:
            fit_y, fit_z = calib_yz
        deploy = {"fitted_on_N": len(fit_y), "target_thresh": args.target_thresh, "calibrators": {}}
        for cls in CALIBRATORS:
            params = fit_final_calibrator(fit_y, fit_z, cls)
            if params is not None:
                deploy["calibrators"][cls.name] = params
        with open(args.save_calibrator, "w", encoding="utf-8") as f:
            json.dump(deploy, f, ensure_ascii=False, indent=2)
        print(f"💾 Saved final calibrator params to {args.save_calibrator}")


if __name__ == "__main__":
    main()
