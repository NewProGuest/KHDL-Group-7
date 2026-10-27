"""Data Reduction cho bài toán IBM HR Attrition (Nhóm 7).

Script đọc `data/processed/hr_attrition_transformed.csv`, thực hiện thu gọn dữ liệu
bằng Feature Selection và so sánh với PCA. Toàn bộ bước chọn đặc trưng chỉ được
fit trên tập TRAIN (và bên trong từng fold khi cross-validation) để tránh rò rỉ dữ liệu.

Quy trình:
  1. Chia train/test (stratified, mặc định 80/20) -> test chỉ dùng để báo cáo cuối.
  2. Trong `rank_features` (chỉ nhìn tập đưa vào):
       a. Loại cột trùng lặp hoàn toàn (vd. Gender_Female/Gender_Male).
       b. Loại biến số liên tục tương quan cao (> ngưỡng), giữ biến có Mutual Information cao hơn.
       c. Xếp hạng bằng 4 phương pháp: Mutual Information, Random Forest importance,
          hệ số L1 Logistic Regression, RFE (Logistic Regression).
       d. Hạng đồng thuận = trung bình hạng của 4 phương pháp.
  3. Stratified K-fold CV (chọn đặc trưng lại trong mỗi fold) cho nhiều giá trị top-k,
     so sánh với "dùng toàn bộ đặc trưng" và PCA. Chọn k nhỏ nhất theo quy tắc one-standard-error.
  4. Fit lại trên toàn bộ train, chọn top-k, đánh giá trên test, xuất báo cáo và dữ liệu đã thu gọn.

Chạy từ thư mục gốc repo:
    python src/reduction/feature_selection.py
    python src/reduction/feature_selection.py --k-grid 10 15 20 25 30 --folds 5 --no-plots
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET = "Attrition"
ID_COL = "EmployeeNumber"
METHOD_COLS = ["mutual_info", "rf_importance", "l1_abs_coef", "rfe_rank"]


# --------------------------------------------------------------------------- #
# Chọn đặc trưng
# --------------------------------------------------------------------------- #
def drop_exact_duplicates(X: pd.DataFrame, thr: float = 0.999) -> tuple[pd.DataFrame, list[dict]]:
    """Loại cột có |corr| >= thr với một cột đã giữ trước đó (vd. dummy của biến nhị phân)."""
    corr = X.corr().abs()
    kept: list[str] = []
    log: list[dict] = []
    # ưu tiên giữ cột "_Yes" (vd. OverTime_Yes) vì dễ diễn giải hơn cột "_No"
    for col in sorted(X.columns, key=lambda c: not c.endswith("_Yes")):
        twin = next((k for k in kept if corr.loc[col, k] >= thr), None)
        if twin is None:
            kept.append(col)
        else:
            log.append({"dropped": col, "kept": twin, "abs_corr": round(float(corr.loc[col, twin]), 3),
                        "reason": "cột trùng lặp (tương quan ~1)"})
    return X[[c for c in X.columns if c in kept]], log


def prune_correlated(X: pd.DataFrame, mi: pd.Series, thr: float) -> tuple[list[str], list[dict]]:
    """Với các biến số (không phải dummy 0/1) có |corr| > thr, loại biến có MI thấp hơn.

    Dummy 0/1 của biến phân loại không bị loại ở bước này vì tương quan giữa các dummy
    (vd. Department_Sales vs Department_R&D) là hệ quả tự nhiên của one-hot encoding.
    """
    numeric = [c for c in X.columns if X[c].nunique() > 2]
    corr = X[numeric].corr().abs()
    pairs = [(a, b, float(corr.loc[a, b]))
             for i, a in enumerate(numeric) for b in numeric[i + 1:] if corr.loc[a, b] > thr]
    pairs.sort(key=lambda t: -t[2])
    dropped: dict[str, dict] = {}
    for a, b, c in pairs:
        if a in dropped or b in dropped:
            continue
        loser, winner = (a, b) if mi[a] < mi[b] else (b, a)
        dropped[loser] = {"dropped": loser, "kept": winner, "abs_corr": round(c, 3),
                          "reason": f"tương quan > {thr} với {winner}, MI thấp hơn"}
    kept = [c for c in X.columns if c not in dropped]
    return kept, list(dropped.values())


def rank_features(X: pd.DataFrame, y: pd.Series, seed: int = 42, corr_thr: float = 0.8,
                  prune: bool = True, vote_top: int = 20) -> tuple[pd.DataFrame, list[dict]]:
    """Trả về bảng xếp hạng đồng thuận (index = tên đặc trưng) và nhật ký các cột bị loại."""
    X, drop_log = drop_exact_duplicates(X)
    mi = pd.Series(mutual_info_classif(X, y, random_state=seed), index=X.columns)
    if prune:
        kept, corr_log = prune_correlated(X, mi, corr_thr)
        drop_log += corr_log
        X, mi = X[kept], mi[kept]

    Xs = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns, index=X.index)

    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1)
    rf.fit(X, y)

    with warnings.catch_warnings():
        # `penalty="l1"` bị cảnh báo deprecated ở scikit-learn >= 1.8 nhưng vẫn chạy đúng.
        warnings.simplefilter("ignore")
        l1 = LogisticRegression(penalty="l1", solver="liblinear", C=0.1,
                                class_weight="balanced", random_state=seed).fit(Xs, y)
        rfe = RFE(LogisticRegression(max_iter=2000, class_weight="balanced"),
                  n_features_to_select=1, step=1).fit(Xs, y)

    df = pd.DataFrame({
        "mutual_info": mi,
        "rf_importance": pd.Series(rf.feature_importances_, index=X.columns),
        "l1_abs_coef": pd.Series(np.abs(l1.coef_[0]), index=X.columns),
        "rfe_rank": pd.Series(rfe.ranking_, index=X.columns),
    })
    ranks = pd.DataFrame({
        "mutual_info": df["mutual_info"].rank(ascending=False),
        "rf_importance": df["rf_importance"].rank(ascending=False),
        "l1_abs_coef": df["l1_abs_coef"].rank(ascending=False),  # hệ số = 0 -> đồng hạng cuối
        "rfe_rank": df["rfe_rank"].astype(float),
    })
    df["mean_rank"] = ranks.mean(axis=1).round(2)
    df["votes_top%d" % vote_top] = (ranks <= vote_top).sum(axis=1)
    df["l1_selected"] = df["l1_abs_coef"] > 0
    df = df.sort_values(["mean_rank", "rf_importance"], ascending=[True, False])
    df.insert(0, "consensus_rank", np.arange(1, len(df) + 1))
    return df, drop_log


# --------------------------------------------------------------------------- #
# Đánh giá
# --------------------------------------------------------------------------- #
def make_models(seed: int) -> dict:
    return {
        "LogReg": make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2000, class_weight="balanced")),
        "RandomForest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                               random_state=seed, n_jobs=-1),
    }


def eval_models(Xtr, ytr, Xva, yva, seed: int) -> dict[str, tuple[float, float]]:
    out = {}
    for name, model in make_models(seed).items():
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xva)[:, 1]
        out[name] = (roc_auc_score(yva, p), average_precision_score(yva, p))
    return out


def eval_pca(Xtr, ytr, Xva, yva, var: float, seed: int) -> tuple[dict, int]:
    scaler = StandardScaler().fit(Xtr)
    pca = PCA(n_components=var, svd_solver="full").fit(scaler.transform(Xtr))
    Ztr, Zva = pca.transform(scaler.transform(Xtr)), pca.transform(scaler.transform(Xva))
    return eval_models(Ztr, ytr, Zva, yva, seed), int(pca.n_components_)


def cross_validate(X, y, k_grid, folds, seed, corr_thr, prune) -> pd.DataFrame:
    """CV với việc chọn đặc trưng được làm lại trong từng fold."""
    rows = []
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(skf.split(X, y), start=1):
        Xtr, Xva, ytr, yva = X.iloc[tr], X.iloc[va], y.iloc[tr], y.iloc[va]
        ranking, _ = rank_features(Xtr, ytr, seed, corr_thr, prune)
        order = list(ranking.index)

        def add(setting, n_feat, res):
            for model, (auc, ap) in res.items():
                rows.append({"fold": fold, "setting": setting, "n_features": n_feat,
                             "model": model, "roc_auc": auc, "pr_auc": ap})

        add("all", X.shape[1], eval_models(Xtr, ytr, Xva, yva, seed))
        for k in k_grid:
            cols = order[:min(k, len(order))]
            add(f"top{k}", len(cols), eval_models(Xtr[cols], ytr, Xva[cols], yva, seed))
        for var in (0.90, 0.95):
            res, n = eval_pca(Xtr, ytr, Xva, yva, var, seed)
            add(f"pca{int(var * 100)}", n, res)
        print(f"  fold {fold}/{folds} xong", flush=True)
    return pd.DataFrame(rows)


def summarize_cv(cv: pd.DataFrame) -> pd.DataFrame:
    g = cv.groupby(["setting", "model"], sort=False)
    s = g.agg(n_features=("n_features", "mean"),
              roc_auc_mean=("roc_auc", "mean"), roc_auc_std=("roc_auc", "std"),
              pr_auc_mean=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std")).round(4).reset_index()
    s["n_features"] = s["n_features"].round(1)
    return s


def pick_k(cv: pd.DataFrame, k_grid: list[int]) -> tuple[int, pd.DataFrame]:
    """Quy tắc one-standard-error: k nhỏ nhất có AUC (TB 2 mô hình) >= best - SE."""
    per_fold = (cv[cv["setting"].isin([f"top{k}" for k in k_grid])]
                .groupby(["setting", "fold"])["roc_auc"].mean().unstack("fold"))
    stats = pd.DataFrame({"mean_auc": per_fold.mean(axis=1),
                          "se": per_fold.std(axis=1, ddof=1) / np.sqrt(per_fold.shape[1])})
    stats["k"] = [int(s[3:]) for s in stats.index]
    stats = stats.sort_values("k")
    best = stats["mean_auc"].idxmax()
    threshold = stats.loc[best, "mean_auc"] - stats.loc[best, "se"]
    k_sel = int(stats[stats["mean_auc"] >= threshold]["k"].min())
    return k_sel, stats


# --------------------------------------------------------------------------- #
# Xuất kết quả
# --------------------------------------------------------------------------- #
def make_plots(ranking, stats, k_sel, outdir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Bỏ qua biểu đồ (chưa cài matplotlib).")
        return
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    top = ranking.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(top.index, top["mean_rank"], color="#4C78A8")
    ax.set_xlabel("Hạng trung bình của 4 phương pháp (càng thấp càng quan trọng)")
    ax.set_title("Top 20 đặc trưng theo hạng đồng thuận")
    fig.tight_layout()
    fig.savefig(figdir / "feature_ranking_top20.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(stats["k"], stats["mean_auc"], yerr=stats["se"], marker="o", capsize=3)
    ax.axvline(k_sel, color="red", ls="--", label=f"k được chọn = {k_sel}")
    ax.set_xlabel("Số đặc trưng giữ lại (k)")
    ax.set_ylabel("ROC-AUC CV (TB LogReg & RF)")
    ax.set_title("ROC-AUC theo số đặc trưng")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "auc_vs_k.png", dpi=150)
    plt.close(fig)


def write_summary(path: Path, ctx: dict) -> None:
    L = []
    L.append("BÁO CÁO DATA REDUCTION - IBM HR ATTRITION (NHÓM 7)")
    L.append("=" * 54)
    L.append(f"Dữ liệu: {ctx['input']}  ({ctx['n_rows']} dòng, {ctx['n_feat']} đặc trưng đầu vào)")
    L.append(f"Chia dữ liệu: train {ctx['n_train']} / test {ctx['n_test']} (stratified, seed={ctx['seed']})")
    L.append(f"CV: Stratified {ctx['folds']}-fold, chọn đặc trưng lại trong từng fold\n")
    L.append("1. CỘT BỊ LOẠI TRƯỚC KHI XẾP HẠNG (trên tập train)")
    for d in ctx["drop_log"]:
        L.append(f"   - {d['dropped']}  (giữ {d['kept']}, |corr|={d['abs_corr']}): {d['reason']}")
    if not ctx["drop_log"]:
        L.append("   (không có)")
    L.append("\n2. KẾT QUẢ CV (trung bình ± độ lệch chuẩn qua các fold)")
    L.append(ctx["cv_table"].to_string(index=False))
    L.append(f"\n   Quy tắc one-standard-error -> chọn k = {ctx['k_sel']} đặc trưng.")
    L.append(f"\n3. {ctx['k_sel']} ĐẶC TRƯNG ĐƯỢC CHỌN (theo hạng đồng thuận, fit trên toàn bộ train)")
    for i, f in enumerate(ctx["selected"], 1):
        L.append(f"   {i:>2}. {f}")
    L.append("\n4. ĐÁNH GIÁ TRÊN TẬP TEST (giữ riêng, chỉ dùng một lần)")
    L.append(ctx["test_table"].to_string(index=False))
    L.append(f"\nGiảm từ {ctx['n_feat']} xuống {ctx['k_sel']} đặc trưng "
             f"({100 * (1 - ctx['k_sel'] / ctx['n_feat']):.0f}% số cột).")
    L.append("\nLƯU Ý:")
    L.append(" - Khuyến nghị dùng Feature Selection thay vì PCA vì giữ được ý nghĩa của từng biến,")
    L.append("   phù hợp mục tiêu phân tích nguyên nhân nghỉ việc.")
    L.append(" - File hr_attrition_reduced.csv chứa toàn bộ dòng với các cột đã chọn; danh sách cột được")
    L.append("   chọn chỉ dựa trên tập train. Khi huấn luyện mô hình hãy chia train/test lại như trong script.")
    L.append(" - Các scaler/encoder trong transform_data.py được fit trên toàn bộ dữ liệu (xem README, mục 6).")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=REPO_ROOT / "data/processed/hr_attrition_transformed.csv")
    ap.add_argument("--reports-dir", type=Path, default=REPO_ROOT / "data/reports")
    ap.add_argument("--processed-dir", type=Path, default=REPO_ROOT / "data/processed")
    ap.add_argument("--k-grid", type=int, nargs="+", default=[5, 10, 15, 20, 25, 30])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--corr-threshold", type=float, default=0.8)
    ap.add_argument("--no-prune", action="store_true", help="không loại biến số tương quan cao")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    y = df[TARGET].astype(int)
    X = df.drop(columns=[c for c in (TARGET, ID_COL) if c in df.columns])
    print(f"Đọc {args.input.name}: {df.shape[0]} dòng, {X.shape[1]} đặc trưng, "
          f"tỷ lệ Attrition = {y.mean():.3f}")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=args.test_size, stratify=y,
                                          random_state=args.seed)
    prune = not args.no_prune
    k_grid = sorted(set(args.k_grid))

    print(f"Cross-validation ({args.folds} folds)...")
    cv = cross_validate(Xtr, ytr, k_grid, args.folds, args.seed, args.corr_threshold, prune)
    cv_table = summarize_cv(cv)
    cv_table["roc_auc"] = cv_table.apply(lambda r: f"{r.roc_auc_mean:.3f} ± {r.roc_auc_std:.3f}", axis=1)
    cv_table["pr_auc"] = cv_table.apply(lambda r: f"{r.pr_auc_mean:.3f} ± {r.pr_auc_std:.3f}", axis=1)
    k_sel, k_stats = pick_k(cv, k_grid)
    print(f"Chọn k = {k_sel} (one-standard-error rule)")

    ranking, drop_log = rank_features(Xtr, ytr, args.seed, args.corr_threshold, prune)
    k_sel = min(k_sel, len(ranking))
    selected = list(ranking.index[:k_sel])

    # Đánh giá cuối trên test
    test_rows = []

    def add_test(setting, n, res):
        for model, (auc, ap) in res.items():
            test_rows.append({"setting": setting, "n_features": n, "model": model,
                              "roc_auc": round(auc, 4), "pr_auc": round(ap, 4)})

    add_test("all", X.shape[1], eval_models(Xtr, ytr, Xte, yte, args.seed))
    add_test(f"top{k_sel}", k_sel, eval_models(Xtr[selected], ytr, Xte[selected], yte, args.seed))
    res, n_pca = eval_pca(Xtr, ytr, Xte, yte, 0.95, args.seed)
    add_test("pca95", n_pca, res)
    test_table = pd.DataFrame(test_rows)

    # Ghi file
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    ranking.round(5).to_csv(args.reports_dir / "feature_ranking.csv", encoding="utf-8")
    pd.DataFrame(drop_log).to_csv(args.reports_dir / "feature_dropped.csv", index=False, encoding="utf-8")
    cv_table.drop(columns=["roc_auc", "pr_auc"]).to_csv(args.reports_dir / "feature_selection_cv.csv",
                                                       index=False, encoding="utf-8")
    test_table.to_csv(args.reports_dir / "feature_selection_test.csv", index=False, encoding="utf-8")
    (args.reports_dir / "selected_features.json").write_text(
        json.dumps({"k": k_sel, "seed": args.seed, "features": selected}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    keep = [c for c in (ID_COL, TARGET) if c in df.columns] + selected
    df[keep].to_csv(args.processed_dir / "hr_attrition_reduced.csv", index=False, encoding="utf-8")

    summary_tbl = cv_table[["setting", "model", "n_features", "roc_auc", "pr_auc"]]
    write_summary(args.reports_dir / "reduction_summary.txt", {
        "input": str(args.input.relative_to(REPO_ROOT)) if args.input.is_relative_to(REPO_ROOT) else str(args.input),
        "n_rows": df.shape[0], "n_feat": X.shape[1], "n_train": len(Xtr), "n_test": len(Xte),
        "seed": args.seed, "folds": args.folds, "drop_log": drop_log, "cv_table": summary_tbl,
        "k_sel": k_sel, "selected": selected, "test_table": test_table,
    })
    if not args.no_plots:
        make_plots(ranking, k_stats, k_sel, args.reports_dir)

    print("\nĐặc trưng được chọn:", ", ".join(selected))
    print("\nTest set:\n", test_table.to_string(index=False))
    print(f"\nĐã ghi báo cáo vào {args.reports_dir} và dữ liệu thu gọn vào "
          f"{args.processed_dir / 'hr_attrition_reduced.csv'}")


if __name__ == "__main__":
    main()
