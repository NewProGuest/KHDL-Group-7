"""Data Reduction cho bộ dữ liệu IBM HR Attrition (Issue #9).

Quy trình:
  B1. Chia train/test (stratified) TRƯỚC khi chọn đặc trưng để tránh rò rỉ dữ liệu.
  B2. Lọc thô trên train: bỏ định danh, biến phương sai ~0, biến dummy dư thừa,
      và với mỗi cặp biến tương quan cao (|r| >= 0.85) giữ biến liên hệ mạnh hơn với Attrition.
  B3. Xếp hạng đặc trưng bằng 5 phương pháp: ANOVA F, Mutual Information,
      L1-Logistic, Random Forest importance, RFE (Logistic).
  B4. Bình chọn: giữ đặc trưng lọt top-K của ít nhất MIN_VOTES phương pháp.
  B5. PCA để so sánh (giữ >= 90% phương sai).
  B6. So sánh các phương án (đầy đủ / sau lọc / bình chọn / PCA) bằng CV trên train
      (bước chọn đặc trưng nằm TRONG pipeline) và đánh giá lần cuối trên test.

Đầu ra:
  data/processed/hr_attrition_reduced.csv
  data/processed/hr_attrition_pca.csv
  data/reports/feature_ranking_report.csv
  data/reports/reduction_comparison_report.csv
  data/reports/reduction_summary.json
  data/reports/figures/*.png
"""
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "data" / "processed" / "hr_attrition_transformed.csv"
OUT_DATA = ROOT / "data" / "processed"
OUT_REPORT = ROOT / "data" / "reports"
OUT_FIG = OUT_REPORT / "figures"

SEED = 42
TEST_SIZE = 0.2
CORR_THRESHOLD = 0.85
MIN_VARIANCE = 0.02      # dummy có < ~2% quan sát = 1 gần như hằng
TOP_K = 20               # mỗi phương pháp bình chọn cho top-K đặc trưng
MIN_VOTES = 3            # cần >= 3/5 phương pháp đồng thuận
PCA_VARIANCE = 0.90
ID_COLS = ["EmployeeNumber"]
TARGET = "Attrition"


# ---------------------------------------------------------------- tiện ích
def correlation_filter(X: pd.DataFrame, y: pd.Series, threshold: float):
    """Với mỗi cặp |r| >= threshold, loại biến có |corr với y| thấp hơn."""
    corr = X.corr().abs()
    target_corr = X.apply(lambda s: abs(np.corrcoef(s, y)[0, 1])).fillna(0)
    dropped = {}
    cols = list(X.columns)
    for i, a in enumerate(cols):
        if a in dropped:
            continue
        for b in cols[i + 1:]:
            if b in dropped or a in dropped:
                continue
            if corr.loc[a, b] >= threshold:
                loser, winner = (a, b) if target_corr[a] < target_corr[b] else (b, a)
                dropped[loser] = (winner, float(corr.loc[a, b]))
    return dropped


def rank_series(scores: pd.Series) -> pd.Series:
    """Hạng 1 = quan trọng nhất."""
    return scores.rank(ascending=False, method="min")


def score_features(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Chấm điểm đặc trưng bằng 5 phương pháp (chỉ dùng dữ liệu train)."""
    cols = X.columns
    out = pd.DataFrame(index=cols)

    out["anova_f"] = pd.Series(f_classif(X, y)[0], index=cols)
    out["mutual_info"] = pd.Series(
        mutual_info_classif(X, y, random_state=SEED, n_neighbors=5), index=cols
    )

    l1 = LogisticRegression(
        penalty="l1", solver="liblinear", C=0.1, class_weight="balanced",
        random_state=SEED, max_iter=2000,
    ).fit(X, y)
    out["l1_logistic_abs_coef"] = pd.Series(np.abs(l1.coef_[0]), index=cols)

    rf = RandomForestClassifier(
        n_estimators=500, min_samples_leaf=5, class_weight="balanced_subsample",
        random_state=SEED, n_jobs=-1,
    ).fit(X, y)
    out["rf_importance"] = pd.Series(rf.feature_importances_, index=cols)

    rfe = RFE(
        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED),
        n_features_to_select=TOP_K, step=1,
    ).fit(X, y)
    # ranking_ = 1 là được giữ; đảo dấu để "điểm cao = tốt"
    out["rfe_score"] = pd.Series(-rfe.ranking_.astype(float), index=cols)
    return out


def vote(scores: pd.DataFrame, top_k: int) -> pd.DataFrame:
    report = scores.copy()
    votes = pd.Series(0, index=scores.index)
    for col in scores.columns:
        report[f"rank_{col}"] = rank_series(scores[col])
        selected = report[f"rank_{col}"] <= top_k
        report[f"top{top_k}_{col}"] = selected.astype(int)
        votes += selected.astype(int)
    report["votes"] = votes
    report["mean_rank"] = report[[c for c in report if c.startswith("rank_")]].mean(axis=1)
    return report.sort_values(["votes", "mean_rank"], ascending=[False, True])


class ColumnKeeper(BaseEstimator, TransformerMixin):
    """Giữ lại danh sách cột cố định."""
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns]


class VotingSelector(BaseEstimator, TransformerMixin):
    """Chạy toàn bộ B2-B4 trong từng fold để CV không bị rò rỉ."""
    def __init__(self, top_k=TOP_K, min_votes=MIN_VOTES):
        self.top_k = top_k
        self.min_votes = min_votes

    def fit(self, X, y):
        dropped = correlation_filter(X, y, CORR_THRESHOLD)
        kept = [c for c in X.columns if c not in dropped and X[c].var() >= MIN_VARIANCE]
        report = vote(score_features(X[kept], y), self.top_k)
        self.selected_ = report.index[report["votes"] >= self.min_votes].tolist()
        return self

    def transform(self, X):
        return X[self.selected_]


def evaluate_cv(name, steps, X, y, n_features_fn=None):
    models = {
        "LogReg": LogisticRegression(class_weight="balanced", max_iter=3000, random_state=SEED),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample",
            random_state=SEED, n_jobs=-1),
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rows = []
    for mname, model in models.items():
        pipe = Pipeline(steps + [("model", model)])
        res = cross_validate(
            pipe, X, y, cv=cv, n_jobs=1,
            scoring={"auc": "roc_auc", "recall": "recall", "f1": "f1", "precision": "precision"},
        )
        rows.append({
            "scheme": name, "model": mname,
            "cv_auc": res["test_auc"].mean(), "cv_auc_std": res["test_auc"].std(),
            "cv_recall": res["test_recall"].mean(),
            "cv_precision": res["test_precision"].mean(),
            "cv_f1": res["test_f1"].mean(),
        })
    return rows


def test_metrics(model, Xtr, ytr, Xte, yte):
    model = clone(model).fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:, 1]
    pred = model.predict(Xte)
    return {
        "test_auc": roc_auc_score(yte, proba),
        "test_recall": recall_score(yte, pred),
        "test_precision": precision_score(yte, pred),
        "test_f1": f1_score(yte, pred),
    }


# ---------------------------------------------------------------- chính
def main() -> None:
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(INPUT_PATH)
    y = data[TARGET]
    X_all = data.drop(columns=[TARGET] + ID_COLS)
    ids = data[ID_COLS]

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    print(f"Tổng: {X_all.shape} | train {X_train.shape} | test {X_test.shape}")
    print(f"Tỷ lệ nghỉ việc train/test: {y_train.mean():.3f} / {y_test.mean():.3f}")

    # ---- B2: lọc thô
    low_var = [c for c in X_train.columns if X_train[c].var() < MIN_VARIANCE]
    corr_dropped = correlation_filter(X_train.drop(columns=low_var), y_train, CORR_THRESHOLD)
    removed = {c: "phương sai ~ 0" for c in low_var}
    removed.update({c: f"tương quan {r:.2f} với {w}" for c, (w, r) in corr_dropped.items()})
    kept_after_filter = [c for c in X_train.columns if c not in removed]
    print(f"\nB2 - Sau lọc thô: {len(X_train.columns)} -> {len(kept_after_filter)} đặc trưng")
    for c, why in removed.items():
        print(f"   bỏ {c}: {why}")

    # ---- B3 + B4: xếp hạng và bình chọn
    scores = score_features(X_train[kept_after_filter], y_train)
    ranking = vote(scores, TOP_K)
    selected = ranking.index[ranking["votes"] >= MIN_VOTES].tolist()
    print(f"\nB4 - Đặc trưng được chọn (>= {MIN_VOTES}/5 phiếu): {len(selected)}")
    for c in selected:
        print(f"   {c}  (phiếu={int(ranking.loc[c, 'votes'])}, hạng TB={ranking.loc[c, 'mean_rank']:.1f})")

    ranking.insert(0, "status", np.where(ranking.index.isin(selected), "selected", "dropped_by_vote"))
    removed_df = pd.DataFrame(
        {"status": "removed_by_filter", "reason": pd.Series(removed)}
    )
    ranking_out = pd.concat([ranking.assign(reason=""), removed_df], sort=False)
    ranking_out.index.name = "feature"
    ranking_out.to_csv(OUT_REPORT / "feature_ranking_report.csv", encoding="utf-8-sig")

    # ---- B5: PCA (fit trên train, giữ >= 90% phương sai)
    pca_full = PCA(random_state=SEED).fit(X_train)
    cum = np.cumsum(pca_full.explained_variance_ratio_)
    n_pca = int(np.searchsorted(cum, PCA_VARIANCE) + 1)
    print(f"\nB5 - PCA cần {n_pca} thành phần để giữ {PCA_VARIANCE:.0%} phương sai (gốc: {X_train.shape[1]})")

    # ---- B6: so sánh
    schemes = {
        "A. Đầy đủ (không giảm)": [("keep", ColumnKeeper(list(X_train.columns)))],
        "B. Sau lọc thô": [("keep", ColumnKeeper(kept_after_filter))],
        "C. Bình chọn 5 phương pháp (đề xuất)": [("select", VotingSelector())],
        f"D. PCA ({PCA_VARIANCE:.0%} phương sai)": [("pca", PCA(n_components=PCA_VARIANCE, random_state=SEED))],
    }
    rows = []
    for name, steps in schemes.items():
        print(f"CV: {name}")
        rows += evaluate_cv(name, steps, X_train, y_train)
    comp = pd.DataFrame(rows)

    # đánh giá trên test (chọn đặc trưng chỉ dựa trên train)
    n_feats = {
        "A. Đầy đủ (không giảm)": X_train.shape[1],
        "B. Sau lọc thô": len(kept_after_filter),
        "C. Bình chọn 5 phương pháp (đề xuất)": len(selected),
        f"D. PCA ({PCA_VARIANCE:.0%} phương sai)": n_pca,
    }
    test_rows = []
    for name, steps in schemes.items():
        for mname, model in {
            "LogReg": LogisticRegression(class_weight="balanced", max_iter=3000, random_state=SEED),
            "RandomForest": RandomForestClassifier(
                n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample",
                random_state=SEED, n_jobs=-1),
        }.items():
            pipe = Pipeline(steps + [("model", model)])
            m = test_metrics(pipe, X_train, y_train, X_test, y_test)
            test_rows.append({"scheme": name, "model": mname, "n_features": n_feats[name], **m})
    comp = comp.merge(pd.DataFrame(test_rows), on=["scheme", "model"])
    comp.round(4).to_csv(OUT_REPORT / "reduction_comparison_report.csv", index=False, encoding="utf-8-sig")
    print("\n", comp.round(3).to_string(index=False))

    # ---- xuất dữ liệu đã thu gọn (toàn bộ 1470 dòng)
    reduced = pd.concat([ids, X_all[selected], y], axis=1)
    reduced.to_csv(OUT_DATA / "hr_attrition_reduced.csv", index=False)
    pca_final = PCA(n_components=n_pca, random_state=SEED).fit(X_train)
    pcs = pd.DataFrame(
        pca_final.transform(X_all), columns=[f"PC{i+1}" for i in range(n_pca)]
    )
    pd.concat([ids, pcs, y], axis=1).to_csv(OUT_DATA / "hr_attrition_pca.csv", index=False)

    # ---- hình
    top = ranking.loc[selected].sort_values("mean_rank", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(top) + 1.5))
    ax.barh(top.index, top["votes"], color="#2b6cb0")
    ax.set_xlabel("Số phương pháp bình chọn (tối đa 5)")
    ax.set_title("Đặc trưng được chọn và số phiếu")
    plt.tight_layout()
    fig.savefig(OUT_FIG / "selected_features_votes.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(cum) + 1), cum, marker="o", ms=3)
    ax.axhline(PCA_VARIANCE, color="r", ls="--", label=f"{PCA_VARIANCE:.0%}")
    ax.axvline(n_pca, color="gray", ls=":")
    ax.set_xlabel("Số thành phần chính")
    ax.set_ylabel("Phương sai tích lũy")
    ax.set_title("PCA - phương sai tích lũy")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUT_FIG / "pca_cumulative_variance.png", dpi=150)
    plt.close(fig)

    cmp_lr = comp[comp["model"] == "LogReg"].set_index("scheme")
    fig, ax = plt.subplots(figsize=(8, 4))
    cmp_lr[["cv_auc", "test_auc"]].plot.bar(ax=ax, rot=15)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("AUC-ROC")
    ax.set_title("So sánh phương án giảm chiều (Logistic Regression)")
    plt.tight_layout()
    fig.savefig(OUT_FIG / "scheme_comparison_auc.png", dpi=150)
    plt.close(fig)

    summary = {
        "n_features_original": int(X_train.shape[1]),
        "n_features_after_filter": len(kept_after_filter),
        "n_features_selected": len(selected),
        "reduction_pct": round(100 * (1 - len(selected) / X_train.shape[1]), 1),
        "pca_components_for_90pct_variance": n_pca,
        "removed_by_filter": removed,
        "selected_features": selected,
        "params": {
            "seed": SEED, "test_size": TEST_SIZE, "corr_threshold": CORR_THRESHOLD,
            "min_variance": MIN_VARIANCE, "top_k": TOP_K, "min_votes": MIN_VOTES,
        },
    }
    (OUT_REPORT / "reduction_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nĐã lưu dữ liệu thu gọn, báo cáo và hình.")


if __name__ == "__main__":
    main()
