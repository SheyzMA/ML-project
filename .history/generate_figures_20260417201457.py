"""
generate_figures.py
Regenerates all report figures from actual run results.
Run from the project root: python generate_figures.py --data_path data/features.npz
"""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

np.random.seed(100)

# ── data helpers ────────────────────────────────────────────────────────────────

def load_and_split(data_path):
    d = np.load(data_path, allow_pickle=True)
    Xtr, Xte = d["xtrain"], d["xtest"]
    yr_tr, yr_te = d["ytrainreg"],  d["ytestreg"]
    yc_tr, yc_te = d["ytrainclassif"], d["ytestclassif"]
    N = Xtr.shape[0]
    idx = np.random.permutation(N)
    split = int(0.7 * N)
    tr, val = idx[:split], idx[split:]
    return (Xtr[tr], Xtr[val], Xte,
            yr_tr[tr], yr_tr[val], yr_te,
            yc_tr[tr], yc_tr[val], yc_te)


def normalize(Xtr, Xval, Xte):
    mu  = Xtr.mean(axis=0, keepdims=True)
    sig = Xtr.std(axis=0,  keepdims=True)
    sig[sig == 0] = 1.0
    def _n(X): return (X - mu) / sig
    return _n(Xtr), _n(Xval), _n(Xte)


def add_bias(*arrays):
    return [np.concatenate([np.ones((X.shape[0], 1)), X], axis=1) for X in arrays]


# ── metrics ──────────────────────────────────────────────────────────────────────

def accuracy(pred, gt):
    return np.mean(pred == gt) * 100.0


def mse(pred, gt):
    return np.mean((pred - gt) ** 2)


def macrof1(pred, gt):
    f1s = []
    for c in np.unique(gt):
        tp = np.sum((pred == c) & (gt == c))
        fp = np.sum((pred == c) & (gt != c))
        fn = np.sum((pred != c) & (gt == c))
        if tp == 0:
            f1s.append(0.0)
        else:
            p = tp / (tp + fp)
            r = tp / (tp + fn)
            f1s.append(2 * p * r / (p + r))
    return np.mean(f1s)


# ── method helpers ───────────────────────────────────────────────────────────────

def run_knn_cls(Xtr, ytr, Xval, yval, ks):
    from src.methods.knn import KNN
    accs, f1s = [], []
    for k in ks:
        m = KNN(k=k, task_kind="classification")
        m.fit(Xtr, ytr)
        pred = m.predict(Xval)
        accs.append(accuracy(pred, yval))
        f1s.append(macrof1(pred, yval))
    return np.array(accs), np.array(f1s)


def run_knn_reg(Xtr, ytr, Xval, yval, ks):
    from src.methods.knn import KNN
    mses = []
    for k in ks:
        m = KNN(k=k, task_kind="regression")
        m.fit(Xtr, ytr)
        pred = m.predict(Xval)
        mses.append(mse(pred, yval))
    return np.array(mses)


def run_logreg_sweep(Xtr, ytr, Xval, yval, lrs, iters_list):
    from src.methods.logistic_regression import LogisticRegression
    acc_grid = np.zeros((len(lrs), len(iters_list)))
    f1_grid  = np.zeros((len(lrs), len(iters_list)))
    for i, lr in enumerate(lrs):
        for j, it in enumerate(iters_list):
            m = LogisticRegression(lr=lr, max_iters=it)
            m.fit(Xtr, ytr)
            pred = m.predict(Xval)
            acc_grid[i, j] = accuracy(pred, yval)
            f1_grid[i, j]  = macrof1(pred, yval)
    return acc_grid, f1_grid


def best_logreg_confusion(Xtr, ytr, Xte, yte, lr=0.01, iters=1000):
    from src.methods.logistic_regression import LogisticRegression
    m = LogisticRegression(lr=lr, max_iters=iters)
    m.fit(Xtr, ytr)
    pred = m.predict(Xte)
    C = int(yte.max()) + 1
    cm = np.zeros((C, C), dtype=int)
    for t, p in zip(yte.astype(int), pred.astype(int)):
        cm[t, p] += 1
    return cm, pred


# ── plot A: KNN tuning curves ────────────────────────────────────────────────────

def plot_knn(ks, cls_acc, cls_f1, reg_mse, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))

    ax = axes[0]
    ax.plot(ks, cls_acc, "o-", color="#1f77b4", label="Accuracy (%)")
    ax.set_xlabel("$k$"); ax.set_ylabel("Validation Accuracy (%)", color="#1f77b4")
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    ax2 = ax.twinx()
    ax2.plot(ks, cls_f1, "s--", color="#ff7f0e", label="Macro-F1")
    ax2.set_ylabel("Validation Macro-F1", color="#ff7f0e")
    ax2.tick_params(axis="y", labelcolor="#ff7f0e")
    best_k = ks[int(np.argmax(cls_acc))]
    ax.axvline(best_k, color="gray", lw=0.8, ls=":")
    ax.set_title("KNN — Classification", fontsize=9)
    ax.set_xticks(ks)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="lower right")

    ax = axes[1]
    ax.plot(ks, reg_mse, "o-", color="#2ca02c")
    ax.set_xlabel("$k$"); ax.set_ylabel("Validation MSE")
    best_k_r = ks[int(np.argmin(reg_mse))]
    ax.axvline(best_k_r, color="gray", lw=0.8, ls=":")
    ax.set_title("KNN — Regression", fontsize=9)
    ax.set_xticks(ks)

    fig.tight_layout(pad=0.8)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


# ── plot B: logistic regression heatmap ─────────────────────────────────────────

def plot_logreg_heatmap(acc_grid, f1_grid, lrs, iters_list, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))

    for ax, grid, title, fmt, cmap in zip(
        axes,
        [acc_grid, f1_grid],
        ["Val. Accuracy (%)", "Val. Macro-F1"],
        [".1f", ".3f"],
        ["YlGn", "YlOrRd"],
    ):
        im = ax.imshow(grid, aspect="auto", cmap=cmap,
                       vmin=grid.min() - 0.01, vmax=grid.max() + 0.01)
        ax.set_xticks(range(len(iters_list)))
        ax.set_xticklabels(iters_list, fontsize=8)
        ax.set_yticks(range(len(lrs)))
        ax.set_yticklabels([f"{lr:.0e}" for lr in lrs], fontsize=8)
        ax.set_xlabel("Iterations"); ax.set_ylabel("Learning rate")
        ax.set_title(f"LogReg — {title}", fontsize=9)

        for i in range(len(lrs)):
            for j in range(len(iters_list)):
                ax.text(j, i, format(grid[i, j], fmt),
                    ha="center", va="center", fontsize=7, color="black")

        bi, bj = np.unravel_index(np.argmax(grid), grid.shape)
        ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1,
                                    fill=False, edgecolor="red", lw=2))
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout(pad=0.8)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[saved] {out_path}")


# ── plot C: confusion matrix ─────────────────────────────────────────────────────

def plot_confusion(cm, out_path):
    labels = ["Low", "Medium", "High"]
    fig, ax = plt.subplots(figsize=(3.2, 2.8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks(range(3)); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("LogReg Confusion (Test)", fontsize=9)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=9, color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


# ── plot D: class distribution ───────────────────────────────────────────────────

def plot_class_dist(yc_tr, yc_te, out_path):
    labels = ["Low", "Medium", "High"]
    fig, axes = plt.subplots(1, 2, figsize=(5, 2.5), sharey=True)
    for ax, y, split_name in zip(axes, [yc_tr, yc_te], ["Train", "Test"]):
        counts = [np.sum(y == c) for c in range(3)]
        bars = ax.bar(labels, counts, color=["#4e9af1", "#f4a261", "#e76f51"],
                      edgecolor="white", linewidth=0.5)
        ax.set_title(f"{split_name} ({len(y)})", fontsize=9)
        ax.set_ylabel("Count")
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, cnt + 5,
                    str(cnt), ha="center", va="bottom", fontsize=8)
    fig.suptitle("Class Distribution", fontsize=9, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/features.npz")
    args = parser.parse_args()

    os.makedirs("figures", exist_ok=True)

    (Xtr, Xval, Xte,
     yr_tr, yr_val, yr_te,
     yc_tr, yc_val, yc_te) = load_and_split(args.data_path)

    Xtr_n, Xval_n, Xte_n = normalize(Xtr, Xval, Xte)

    # bias versions for logistic
    Xtr_b, Xval_b, Xte_b = add_bias(Xtr_n, Xval_n, Xte_n)

    ks = [1, 3, 5, 7, 9, 11, 15, 21]
    lrs = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]
    iters_list = [200, 500, 1000, 2000]

    print("Running KNN classification sweep...")
    cls_acc, cls_f1 = run_knn_cls(Xtr_n, yc_tr, Xval_n, yc_val, ks)

    print("Running KNN regression sweep...")
    reg_mse = run_knn_reg(Xtr_n, yr_tr, Xval_n, yr_val, ks)

    print("Running LogReg hyperparameter sweep...")
    lr_grid, f1_grid = run_logreg_sweep(Xtr_b, yc_tr, Xval_b, yc_val, lrs, iters_list)

    print("Computing confusion matrix on test set...")
    # retrain on full train (Xtr + Xval) for test evaluation
    Xfull = np.concatenate([Xtr, Xval], axis=0)
    yfull_c = np.concatenate([yc_tr, yc_val], axis=0)
    mu = Xfull.mean(0, keepdims=True); sig = Xfull.std(0, keepdims=True); sig[sig==0]=1
    Xfull_n = (Xfull - mu) / sig; Xte_n2 = (Xte - mu) / sig
    Xfull_b, Xte_b2 = add_bias(Xfull_n, Xte_n2)
    cm, _ = best_logreg_confusion(Xfull_b, yfull_c, Xte_b2, yc_te, lr=0.3, iters=1000)

    plot_knn(ks, cls_acc, cls_f1, reg_mse, "figures/knn_tuning.png")
    plot_logreg_heatmap(lr_grid, f1_grid, lrs, iters_list, "figures/logreg_heatmap.png")
    plot_confusion(cm, "figures/confusion_matrix.png")
    # Use full training labels (1600), not only the 70% split (1120).
    plot_class_dist(yfull_c, yc_te, "figures/class_distribution.png")

    # print numeric summaries for report
    print("\n=== KNN classification (val) ===")
    for k, a, f in zip(ks, cls_acc, cls_f1):
        print(f"  k={k:2d}  acc={a:.2f}%  f1={f:.4f}")
    print(f"\n=== KNN regression (val) ===")
    for k, m in zip(ks, reg_mse):
        print(f"  k={k:2d}  mse={m:.4f}")
    print("\n=== LogReg grid (val accuracy %) ===")
    header = "lr \\ iters   " + "  ".join(f"{it:5d}" for it in iters_list)
    print(header)
    for i, lr in enumerate(lrs):
        row = f"{lr:.0e}        " + "  ".join(f"{lr_grid[i,j]:5.2f}" for j in range(len(iters_list)))
        print(row)
    print(f"\n=== Confusion matrix (LogReg, test) ===")
    print(cm)


if __name__ == "__main__":
    main()
