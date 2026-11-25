"""Оценка качества модели по сохранённым предсказаниям валидации (GPU не нужна).

Запуск:   python evaluate.py --run d121_g1_f0

Берёт из runs/<run>/ файлы val_probs.npy, val_labels.npy, val_rows.npy (их пишет train.py
на лучшей эпохе) и считает: AUROC/AUPRC по болезням, пороги (Youden, Spec@Sens90, Sens@Spec90),
калибровку (ECE, температурное масштабирование, масштабирование Платта), подгруппы
(пол / возраст / view position) и сводную проверку требований ТЗ.

ВАЖНО: это ВАЛИДАЦИЯ фолда, а не Test. Test оценивается один раз в самом конце.
Результаты кладутся в runs/<run>/eval/.
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import GroupKFold

import config as C
import metrics as M

TARGET_AUROC, TARGET_ECE, GAP_FLAG = 0.80, 0.10, 0.10


# ---------- вспомогательное ----------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def f3(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def safe_mean_auroc(y, p, min_pos=5):
    vals = []
    for k in range(y.shape[1]):
        pos = y[:, k].sum()
        if min_pos <= pos <= len(y) - min_pos:
            vals.append(M.per_class_auroc(y[:, [k]], p[:, [k]])[0])
    return float(np.mean(vals)) if vals else float("nan")


# ---------- рабочие точки ----------
def at_threshold(y, p, thr):
    pred = p >= thr
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    return {"threshold": float(min(thr, 1.0)),
            "sens": tp / max(tp + fn, 1), "spec": tn / max(tn + fp, 1),
            "ppv": tp / max(tp + fp, 1), "npv": tn / max(tn + fn, 1)}


def operating_points(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    youden = thr[int(np.argmax(tpr - fpr))]
    sens90 = thr[np.where(tpr >= 0.90)[0][0]]     # самый высокий порог, где чувствительность >= 90%
    spec90 = thr[np.where(fpr <= 0.10)[0][-1]]    # самый низкий порог, где специфичность >= 90%
    return {"youden": at_threshold(y, p, youden),
            "sens90": at_threshold(y, p, sens90),
            "spec90": at_threshold(y, p, spec90)}


# ---------- калибровка ----------
def fit_temperature(y, p):
    z = logit(p)

    def nll(T):
        q = np.clip(sigmoid(z / T), 1e-7, 1 - 1e-7)
        return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
    return float(minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x)


def fit_platt(y, p):
    lr = LogisticRegression(C=1e6, max_iter=2000).fit(logit(p).reshape(-1, 1), y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def crossfit_calibration(y, p, groups):
    """Параметры подбираются на одной половине пациентов, ECE считается на другой -
    так оценка не завышается тем, что калибровали и проверяли на одних данных."""
    out = {"none": p.copy(), "temp": np.zeros_like(p), "platt": np.zeros_like(p)}
    for tr, te in GroupKFold(2).split(p, groups=groups):
        T = fit_temperature(y[tr], p[tr])
        out["temp"][te] = sigmoid(logit(p[te]) / T)
        for k in range(p.shape[1]):
            a, b = fit_platt(y[tr, k], p[tr, k])
            out["platt"][te, k] = sigmoid(a * logit(p[te, k]) + b)
    return out


def reliability_png(y, curves, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "k--", label="идеал")
    for name, p in curves.items():
        yy, pp = y.ravel(), p.ravel()
        idx = np.clip(np.digitize(pp, np.linspace(0, 1, 11)[1:-1]), 0, 9)
        xs, ys = [], []
        for b in range(10):
            m = idx == b
            if m.sum() >= 20:
                xs.append(pp[m].mean()); ys.append(yy[m].mean())
        ax.plot(xs, ys, marker="o", label=name)
    ax.set_xlabel("предсказанная вероятность"); ax.set_ylabel("доля положительных")
    ax.set_title("Reliability diagram (все болезни вместе)"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ---------- подгруппы ----------
def subgroup(y, p, meta, col, order=None):
    vals = order or sorted(meta[col].dropna().unique())
    rows = []
    for v in vals:
        idx = np.where(meta[col].values == v)[0]
        if len(idx) >= 50:
            rows.append((v, len(idx), safe_mean_auroc(y[idx], p[idx])))
    return rows


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="имя папки в runs/, например d121_g1_f0")
    a = ap.parse_args()
    run = C.RUNS_DIR / a.run
    out = run / "eval"; out.mkdir(exist_ok=True)

    y = np.load(run / "val_labels.npy").astype(np.float64)
    p = np.load(run / "val_probs.npy").astype(np.float64)
    rows = np.load(run / "val_rows.npy")
    meta = pd.read_csv(C.METADATA_CSV).iloc[rows].reset_index(drop=True)
    assert len(y) == len(p) == len(meta), "размеры предсказаний и метаданных не совпали"

    # 1) метрики по болезням
    au, ap_ = M.per_class_auroc(y, p), M.per_class_auprc(y, p)
    mean_au, mean_ap, ece_raw = float(np.nanmean(au)), float(np.nanmean(ap_)), M.mean_ece(y, p)
    per_class = [(d, int(y[:, k].sum()), au[k], ap_[k]) for k, d in enumerate(C.DISEASES)]

    # 2) рабочие точки
    ops = {d: operating_points(y[:, k], p[:, k]) for k, d in enumerate(C.DISEASES) if y[:, k].sum() > 0}

    # 3) калибровка (честная, с кросс-подбором по пациентам)
    cal = crossfit_calibration(y, p, meta["patient_id"].values)
    ece = {k: M.mean_ece(y, v) for k, v in cal.items()}
    T_full = fit_temperature(y, p)
    platt_full = {d: fit_platt(y[:, k], p[:, k]) for k, d in enumerate(C.DISEASES)}
    json.dump({"note": "параметры подобраны на ВСЁМ валидационном фолде; применять к Test без переподбора",
               "temperature": T_full,
               "platt": {d: {"a": a_, "b": b_} for d, (a_, b_) in platt_full.items()}},
              open(out / "calibration.json", "w"), indent=2)
    reliability_png(y, {"до калибровки": cal["none"], "температура": cal["temp"],
                        "Платт по болезням": cal["platt"]}, out / "reliability.png")

    # 4) подгруппы
    sub = {"Пол": subgroup(y, p, meta, "sex"),
           "Возраст": subgroup(y, p, meta, "age_group", ["0-40", "40-60", "60+"]),
           "View position": subgroup(y, p, meta, "view")}

    # ---- отчёт ----
    L = [f"# Оценка модели {a.run} (валидация фолда, не Test)", "",
         f"Снимков: {len(y):,}, пациентов: {meta['patient_id'].nunique():,}. "
         "Метрики основаны на метках NIH (авторазметка NLP, содержит шум) и не равны клинической эффективности.", "",
         "## 1. Сравнение с требованиями ТЗ", ""]
    best_cal = min(("temp", "platt"), key=lambda k: ece[k])
    L.append(md_table(["Требование", "Цель", "Сейчас", "Статус"], [
        ["Средний AUROC", f">= {TARGET_AUROC}", f3(mean_au), "ВЫПОЛНЕНО" if mean_au >= TARGET_AUROC else "НЕ ВЫПОЛНЕНО"],
        ["ECE до калибровки", "<= 0.10", f3(ece_raw), "ВЫПОЛНЕНО" if ece_raw <= TARGET_ECE else "НЕ ВЫПОЛНЕНО"],
        ["ECE после температурного масштабирования", "<= 0.10", f3(ece["temp"]),
         "ВЫПОЛНЕНО" if ece["temp"] <= TARGET_ECE else "НЕ ВЫПОЛНЕНО"],
        ["ECE после масштабирования Платта (по болезням)", "<= 0.10", f3(ece["platt"]),
         "ВЫПОЛНЕНО" if ece["platt"] <= TARGET_ECE else "НЕ ВЫПОЛНЕНО"],
        ["Время инференса", "<= 500 мс", "не измерено (нужен API)", "ОТКРЫТО"],
    ]))
    L += ["", f"Средний AUPRC: {f3(mean_ap)}. Лучший способ калибровки здесь: "
              f"{'температура' if best_cal == 'temp' else 'Платт по болезням'}.", "",
          "## 2. AUROC и AUPRC по болезням", "",
          md_table(["Болезнь", "Положительных", "AUROC", "AUPRC"],
                   [(d, n, f3(x), f3(z)) for d, n, x, z in per_class]), "",
          "## 3. Рабочие точки (пороги)", "",
          "Youden: баланс чувствительности и специфичности. Sens90: порог, при котором чувствительность >= 90% "
          "(скрининг). Spec90: порог, при котором специфичность >= 90% (подтверждающая диагностика).", "",
          md_table(["Болезнь", "Youden: порог", "Sens", "Spec", "Sens90: порог", "Spec@Sens90",
                    "Spec90: порог", "Sens@Spec90", "PPV@Spec90"],
                   [(d, f3(o["youden"]["threshold"]), f3(o["youden"]["sens"]), f3(o["youden"]["spec"]),
                     f3(o["sens90"]["threshold"]), f3(o["sens90"]["spec"]),
                     f3(o["spec90"]["threshold"]), f3(o["spec90"]["sens"]), f3(o["spec90"]["ppv"]))
                    for d, o in ops.items()]), "",
          "## 4. Калибровка", "",
          md_table(["Способ", "ECE"], [["Без калибровки", f3(ece["none"])],
                                       [f"Температура (T на всём фолде = {T_full:.2f})", f3(ece["temp"])],
                                       ["Платт по болезням", f3(ece["platt"])]]), "",
          *(["**Внимание:** температура упёрлась в границу поиска (0.05-20): одного общего масштаба недостаточно, "
             "вероятности смещены (обычно из-за больших pos_weight). Используйте масштабирование Платта по болезням.", ""]
            if T_full >= 19.9 or T_full <= 0.06 else []),
          "ECE считается как среднее по 14 болезням (10 интервалов). Параметры калибровки подбирались на одной "
          "половине пациентов, а ECE считалась на другой (кросс-подбор), чтобы оценка не была завышена. "
          "Подобранные на всём фолде параметры лежат в calibration.json; на Test их надо применять без переподбора. "
          "График: reliability.png.", "",
          "## 5. Подгруппы (средний AUROC)", ""]
    for title, rws in sub.items():
        if not rws:
            continue
        L += [f"**{title}**", "", md_table(["Группа", "N", "Mean AUROC"], [(v, n, f3(m)) for v, n, m in rws])]
        ms = [m for _, _, m in rws if not np.isnan(m)]
        gap = (max(ms) - min(ms)) if len(ms) > 1 else 0.0
        L += ["", f"Разброс между группами: {gap:.3f}. " +
              ("**>= 0.10: по ТЗ нужен разбор причин.**" if gap >= GAP_FLAG else "Меньше порога 0.10 из ТЗ."), ""]
    L += ["## 6. Ограничения", "",
          "- Это один фолд и один запуск: разброс между фолдами обычно порядка 0.005 AUROC.",
          "- Валидация используется и для выбора лучшей эпохи, поэтому оценка слегка оптимистична. "
          "Итоговые цифры нужно получать на Test, один раз.",
          "- У редких болезней (Hernia, Pneumonia) мало положительных примеров, их метрики неустойчивы."]

    (out / "eval_report.md").write_text("\n".join(L), encoding="utf-8")
    pd.DataFrame(per_class, columns=["disease", "n_pos", "auroc", "auprc"]).to_csv(out / "per_class.csv", index=False)
    json.dump({"mean_auroc": mean_au, "mean_auprc": mean_ap, "ece": ece, "temperature": T_full,
               "operating_points": ops}, open(out / "eval_metrics.json", "w"), indent=2)
    print("\n".join(L))
    print(f"\nСохранено в {out}")


if __name__ == "__main__":
    main()
