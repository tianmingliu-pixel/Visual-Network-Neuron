"""
模型表现分析 / Model evaluation for the custom-data task
=======================================================
  · 分类：准确率、混淆矩阵、每类精确率/召回率/F1、最自信的错误样本
  · 回归：R²、MAE、RMSE、预测-真实散点、残差分布、误差最大的样本
  · 特征重要性（置换法）：把某个特征在验证集里随机打乱，看指标下降多少 —— 下降越多说明模型越依赖它
"""
from __future__ import annotations

import math

import numpy as np
import torch


def _r(x, k=4):
    x = float(x)
    return round(x, k) if math.isfinite(x) else None


@torch.no_grad()
def predict(model, X, device, batch=512):
    model.eval()
    outs = []
    for i in range(0, len(X), batch):
        outs.append(model(X[i:i + batch].to(device)).float().cpu())
    return torch.cat(outs) if outs else torch.empty(0)


def _cls_metrics(y, pred, K):
    cm = np.zeros((K, K), np.int64)
    np.add.at(cm, (y, pred), 1)
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return cm, prec, rec, f1


def quick_eval(task, max_val=2000):
    """每秒刷新的轻量评估（预览用）/ lightweight evaluation for live preview."""
    ds = task.dataset
    va = task.va[:max_val]
    out = predict(task.model, task.X[va], task.device)
    y = task.Y[va].numpy()
    if task.regression:
        p = out.squeeze(-1).numpy() * ds.y_std + ds.y_mean
        t = y * ds.y_std + ds.y_mean
        ss = ((t - t.mean()) ** 2).sum()
        r2 = 1 - ((t - p) ** 2).sum() / ss if ss > 0 else 0.0
        k = np.linspace(0, len(t) - 1, min(400, len(t))).astype(int)
        return {"kind": "eval", "task_type": "regression", "r2": _r(r2), "mae": _r(np.abs(t - p).mean()),
                "rmse": _r(np.sqrt(((t - p) ** 2).mean())), "points": [[_r(t[i], 4), _r(p[i], 4)] for i in k],
                "target": ds.target_name}
    pred = out.argmax(-1).numpy()
    K = len(ds.class_names)
    cm, prec, rec, f1 = _cls_metrics(y, pred, K)
    res = {"kind": "eval", "task_type": "classification", "acc": _r((pred == y).mean()), "classes": ds.class_names,
           "confusion": cm.tolist() if K <= 30 else None, "recall": [_r(v) for v in rec], "n_val": int(len(y))}
    if ds.is_image:                                             # 附几张验证图 / a few validation images
        conf = out.softmax(-1).max(-1).values.numpy()
        items = []
        for j in np.linspace(0, len(va) - 1, min(8, len(va))).astype(int):
            img = task.X[va[j]].numpy()
            px = ((img.transpose(1, 2, 0) + 1) * 127.5).clip(0, 255).astype(np.uint8)
            items.append({"w": img.shape[2], "h": img.shape[1], "c": img.shape[0], "px": px.flatten().tolist(),
                          "label": ds.class_names[y[j]], "pred": ds.class_names[pred[j]], "conf": _r(conf[j], 3)})
        res["items"] = items
    return res


def full_eval(task, max_val=3000, max_features=60, seed=0):
    """按需运行的完整分析 / on-demand full analysis."""
    ds = task.dataset
    rng = np.random.default_rng(seed)
    va = task.va[:max_val]
    Xv, yv = task.X[va], task.Y[va].numpy()
    out = predict(task.model, Xv, task.device)
    res = {"task_type": ds.task_type, "n_val": int(len(va))}

    def score(o):
        if task.regression:
            return -float(((o.squeeze(-1).numpy() - yv) ** 2).mean())
        return float((o.argmax(-1).numpy() == yv).mean())

    base = score(out)
    if task.regression:
        p = out.squeeze(-1).numpy() * ds.y_std + ds.y_mean
        t = yv * ds.y_std + ds.y_mean
        err = p - t
        ss = ((t - t.mean()) ** 2).sum()
        res.update(r2=_r(1 - (err ** 2).sum() / ss if ss > 0 else 0), mae=_r(np.abs(err).mean()),
                   rmse=_r(np.sqrt((err ** 2).mean())))
        h, e = np.histogram(err, bins=24)
        res["residual_hist"] = {"counts": h.tolist(), "min": _r(e[0]), "max": _r(e[-1])}
        worst = np.argsort(-np.abs(err))[:8]
        res["worst"] = [{"index": int(va[i]), "target": _r(t[i]), "pred": _r(p[i])} for i in worst]
    else:
        prob = out.softmax(-1).numpy()
        pred = prob.argmax(-1)
        K = len(ds.class_names)
        cm, prec, rec, f1 = _cls_metrics(yv, pred, K)
        res.update(acc=_r((pred == yv).mean()), classes=ds.class_names, confusion=cm.tolist() if K <= 30 else None,
                   per_class=[{"name": ds.class_names[k], "precision": _r(prec[k]), "recall": _r(rec[k]),
                               "f1": _r(f1[k]), "support": int(cm[k].sum())} for k in range(K)])
        wrong = np.where(pred != yv)[0]
        wrong = wrong[np.argsort(-prob[wrong, pred[wrong]])][:8]
        items = []
        for i in wrong:
            it = {"index": int(va[i]), "label": ds.class_names[yv[i]], "pred": ds.class_names[pred[i]],
                  "conf": _r(prob[i, pred[i]], 3)}
            if ds.is_image:
                img = Xv[i].numpy()
                it.update(w=img.shape[2], h=img.shape[1], c=img.shape[0],
                          px=((img.transpose(1, 2, 0) + 1) * 127.5).clip(0, 255).astype(np.uint8).flatten().tolist())
            items.append(it)
        res["worst"] = items
    # 置换特征重要性 / permutation importance (vector inputs only)
    if not ds.is_image:
        D = Xv.shape[1]
        W1 = task.model.layers[0].weight.detach().abs().sum(0).cpu().numpy()
        cand = np.argsort(-W1)[:max_features]
        imp = []
        for j in cand:
            Xp = Xv.clone()
            Xp[:, j] = Xp[rng.permutation(len(Xp)), j]
            imp.append((base - score(predict(task.model, Xp, task.device)), j))
        imp.sort(reverse=True)
        res["importance"] = [{"name": ds.feature_names[j], "drop": _r(d)} for d, j in imp[:20]]
        res["importance_metric"] = "准确率下降" if not task.regression else "MSE 上升（标准化后）"
        if D > max_features:
            res["importance_note"] = f"共 {D} 个特征，按第一层权重大小挑了前 {max_features} 个做置换测试。"
    return res
