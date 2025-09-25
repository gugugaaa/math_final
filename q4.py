import os
import random
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
import xgboost as xgb
import pickle

warnings.filterwarnings("ignore")

# 配置类
class Config:
    SEED = 2025
    DATA_PATH = "data/q4/q4_girl.xlsx"
    FPR_CAP = 0.10
    Z_FLOOR_LOW = 1.5
    ART_DIR = "nipt_girl_xgb_v6_artifacts"
    PIPELINE_PATH = "/mnt/data/girl_xgb_pipeline_v6.pkl"

# 设置随机种子
np.random.seed(Config.SEED)
random.seed(Config.SEED)

# 数据加载
try:
    from template import load_prenatal_data
except ImportError:
    def load_prenatal_data(file_path):
        column_headers = [
            "序号", "孕妇代码", "年龄", "身高", "体重", "末次月经", "IVF妊娠", 
            "检测日期", "检测抽血次数", "检测孕周", "孕妇BMI", "原始读段数", 
            "在参考基因组上比对的比例", "重复读段的比例", "唯一比对的读段数", "GC含量",
            "13号染色体的Z值", "18号染色体的Z值", "21号染色体的Z值", "X染色体的Z值",
            "Unnamed_21", "Unnamed_22", "X染色体浓度", "13号染色体的GC含量", 
            "18号染色体的GC含量", "21号染色体的GC含量", "被过滤掉读段数的比例",
            "染色体的非整倍体", "怀孕次数", "生产次数", "胎儿是否健康"
        ]
        df = pd.read_excel(file_path, header=0, names=column_headers)
        df = df.drop(columns=["Unnamed_21", "Unnamed_22"])
        def parse_gestational_week(week_str):
            if pd.isna(week_str): return None
            week_str = str(week_str).strip()
            if 'w' in week_str:
                parts = week_str.split('w'); weeks = int(parts[0])
                if '+' in parts[1] and parts[1].strip() != '':
                    days = int(parts[1].replace('+', '')); return weeks + days / 7
                else:
                    return weeks
            try:
                return float(week_str)
            except:
                return None
        df['检测孕周'] = df['检测孕周'].apply(parse_gestational_week)
        def process_pregnancy_count(x):
            if pd.isna(x): return x
            try:
                count = int(x); return min(count, 3)
            except:
                return x
        df['怀孕次数'] = df['怀孕次数'].apply(process_pregnancy_count)
        df['染色体的非整倍体'] = df['染色体的非整倍体'].fillna('健康')
        df['Is_T21'] = df['染色体的非整倍体'].apply(lambda x: 1 if 'T21' in str(x) else 0)
        df['Is_T18'] = df['染色体的非整倍体'].apply(lambda x: 1 if 'T18' in str(x) else 0)
        df['Is_T13'] = df['染色体的非整倍体'].apply(lambda x: 1 if 'T13' in str(x) else 0)
        return df

# 特征构建器类
class FeatureBuilder:
    def __init__(self, z_thr=3.0):
        self.z_thr = z_thr
        self.xconc_med = None
        self.lowcov_q_unique = None
        self.lowcov_q_reads = None
        self.feature_names = None
        self.mono = None

    def _base(self, df):
        out = pd.DataFrame(index=df.index)
        base_cols = ["孕妇BMI", "年龄", "检测孕周", "怀孕次数", "生产次数", "检测抽血次数",
                     "原始读段数", "在参考基因组上比对的比例", "重复读段的比例", "唯一比对的读段数", "GC含量",
                     "13号染色体的Z值", "18号染色体的Z值", "21号染色体的Z值", "X染色体的Z值", "X染色体浓度",
                     "13号染色体的GC含量", "18号染色体的GC含量", "21号染色体的GC含量", "被过滤掉读段数的比例"]
        for k in base_cols:
            out[k] = pd.to_numeric(df[k], errors="coerce") if k in df.columns else np.nan
        # 将比例标准化到0-1范围
        for k in ["GC含量", "13号染色体的GC含量", "18号染色体的GC含量", "21号染色体的GC含量",
                  "在参考基因组上比对的比例", "重复读段的比例", "被过滤掉读段数的比例"]:
            if k in out:
                mx = pd.Series(out[k]).dropna().max()
                out[k] = out[k] / 100.0 if (pd.notna(mx) and mx > 1.0) else out[k]
        # 衍生特征
        out["gest_weeks"] = out["检测孕周"]
        out["unique_ratio"] = (out["唯一比对的读段数"] / out["原始读段数"]).replace([np.inf, -np.inf], np.nan)
        out["log_total_reads"] = np.log1p(out["原始读段数"])
        out["log_unique_reads"] = np.log1p(out["唯一比对的读段数"])
        out["eff_ratio"] = (out["在参考基因组上比对的比例"] * (1.0 - out["重复读段的比例"]) * (1.0 - out["被过滤掉读段数的比例"])).astype(float)
        out["gc_all_dev"] = (out["GC含量"] - 0.5).abs()
        out["gc13_dev"] = (out["13号染色体的GC含量"] - 0.5).abs()
        out["gc18_dev"] = (out["18号染色体的GC含量"] - 0.5).abs()
        out["gc21_dev"] = (out["21号染色体的GC含量"] - 0.5).abs()
        out["gc13_d"] = out["13号染色体的GC含量"] - out["GC含量"]
        out["gc18_d"] = out["18号染色体的GC含量"] - out["GC含量"]
        out["gc21_d"] = out["21号染色体的GC含量"] - out["GC含量"]
        # X染色体稳定性
        if self.xconc_med is None:
            x = pd.to_numeric(df.get("X染色体浓度", pd.Series([np.nan]*len(df))), errors="coerce")
            self.xconc_med = float(pd.Series(x).median(skipna=True)) if len(pd.Series(x).dropna()) else 0.0
        xconc = pd.to_numeric(df.get("X染色体浓度", pd.Series([np.nan]*len(df))), errors="coerce")
        out["xconc_dev"] = (xconc - self.xconc_med).abs()
        out["zx"] = out["X染色体的Z值"]
        out["zx_abs"] = out["X染色体的Z值"].abs()
        out["zx_pos"] = out["X染色体的Z值"].clip(lower=0)
        # Z值特征
        out["z13"] = out["13号染色体的Z值"]
        out["z18"] = out["18号染色体的Z值"]
        out["z21"] = out["21号染色体的Z值"]
        out["absZ13"] = out["z13"].abs()
        out["absZ18"] = out["z18"].abs()
        out["absZ21"] = out["z21"].abs()
        out["QZ13"] = out["absZ13"] * out["eff_ratio"].clip(0, 1)
        out["QZ18"] = out["absZ18"] * out["eff_ratio"].clip(0, 1)
        out["QZ21"] = out["absZ21"] * out["eff_ratio"].clip(0, 1)
        out["absZmax"] = np.nanmax(np.abs(out[["z13", "z18", "z21"]].values), axis=1)
        # 质量因子
        out["quality_factor"] = (1.0
            + 5.0 * out["重复读段的比例"].fillna(0) + 5.0 * out["被过滤掉读段数的比例"].fillna(0) + 4.0 * out["gc_all_dev"].fillna(0)
            + 1.5 * (1.0 - out["unique_ratio"].fillna(0)).clip(0, 1)
            + 1.0 * ((0.5 - out["在参考基因组上比对的比例"].fillna(0)).clip(lower=0))
        )
        for h in ["13", "18", "21"]:
            out[f"z{h}_pos"] = out[f"z{h}"].clip(lower=0)
            out[f"z{h}_pos_m"] = (out[f"z{h}"] - self.z_thr).clip(lower=0)
            out[f"z{h}_pos_m_adj"] = out[f"z{h}_pos_m"] / out["quality_factor"].replace(0, np.nan)
        return out

    def fit(self, df_fit):
        tmp = self._base(df_fit.copy())
        uq = tmp["unique_ratio"].dropna()
        ru = tmp["log_unique_reads"].dropna()
        self.lowcov_q_unique = float(np.quantile(uq, 0.10)) if len(uq) else 0.0
        self.lowcov_q_reads = float(np.quantile(ru, 0.10)) if len(ru) else 0.0
        return self

    def transform(self, df):
        X = self._base(df.copy())
        X["lowcov_flag"] = ((X["unique_ratio"] < self.lowcov_q_unique) | (X["log_unique_reads"] < self.lowcov_q_reads)).astype(float)
        feats = [
            "z13_pos", "z18_pos", "z21_pos", "z13_pos_m", "z18_pos_m", "z21_pos_m",
            "z13_pos_m_adj", "z18_pos_m_adj", "z21_pos_m_adj",
            "quality_factor", "gc_all_dev", "gc13_dev", "gc18_dev", "gc21_dev", "zx_abs", "xconc_dev",
            "log_total_reads", "log_unique_reads", "unique_ratio", "eff_ratio",
            "gc13_d", "gc18_d", "gc21_d",
            "z13", "z18", "z21", "QZ13", "QZ18", "QZ21", "absZmax", "zx_pos",
            "孕妇BMI", "年龄", "gest_weeks", "怀孕次数", "生产次数", "检测抽血次数"
        ]
        self.feature_names = feats
        mono = []
        for f in feats:
            if f in {"z13_pos", "z18_pos", "z21_pos", "z13_pos_m", "z18_pos_m", "z21_pos_m",
                     "z13_pos_m_adj", "z18_pos_m_adj", "z21_pos_m_adj",
                     "quality_factor", "gc_all_dev", "gc13_dev", "gc18_dev", "gc21_dev", "zx_abs", "xconc_dev"}:
                mono.append(+1)
            elif f in {"log_total_reads", "log_unique_reads", "unique_ratio", "eff_ratio"}:
                mono.append(-1)
            else:
                mono.append(0)
        self.mono = mono
        return X[feats].astype("float32")

# 恒等校准器
class IdentityCalibrator:
    def predict_proba(self, X):
        probs = np.clip(np.asarray(X).ravel(), 0, 1)
        return np.column_stack([1 - probs, probs])
    
    def __repr__(self):
        return "IdentityCalibrator()"

# 评估函数
def pr_metrics(y, p, name):
    auc = roc_auc_score(y, p)
    ap = average_precision_score(y, p)
    print(f"[{name}] AUC={auc:.6f}  PR-AUC={ap:.6f}")
    return auc, ap

def recall_at_fpr_cap(y_true, p, cap=0.10):
    fpr, tpr, thr = roc_curve(y_true, p)
    best_rec, best_thr, best_fpr = 0.0, 0.5, 1.0
    for fp, tp, t in zip(fpr, tpr, thr):
        if fp <= cap and tp >= best_rec:
            best_rec, best_thr, best_fpr = float(tp), float(t), float(fp)
    return best_rec, best_thr, best_fpr

# 分组分层划分
def stratified_group_split(groups, y, test_size=0.2, val_size=0.2, seed=Config.SEED):
    rng = random.Random(seed)
    gdf = pd.DataFrame({"g": groups, "y": y}).groupby("g")["y"].max()
    pos = [k for k, v in gdf.items() if v == 1]
    neg = [k for k, v in gdf.items() if v == 0]
    def take(lst, frac):
        k = max(1, int(round(len(lst) * frac))) if len(lst) > 0 else 0
        lst = lst.copy(); rng.shuffle(lst); return set(lst[:k])
    te = set(take(pos, test_size) | take(neg, test_size))
    rem_pos = [x for x in pos if x not in te]; rem_neg = [x for x in neg if x not in te]
    vfrac = val_size / max(1e-9, 1.0 - test_size)
    va = set(take(rem_pos, vfrac) | take(rem_neg, vfrac))
    idx = np.arange(len(groups))
    te_idx = idx[np.isin(groups, list(te))]
    va_idx = idx[np.isin(groups, list(va))]
    tr_idx = idx[~np.isin(groups, list(te | va))]
    return tr_idx, va_idx, te_idx

# 分组折叠迭代器
def iter_group_folds(X, y, groups, n_splits=5, seed=Config.SEED):
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return cv.split(X, y, groups)

# 构建XGBoost模型
def build_xgb(params, mono):
    base = {
        "objective": "binary:logistic",
        "eval_metric": ["auc", "aucpr"],
        "seed": Config.SEED,
        "tree_method": "hist",
        "enable_categorical": False,
        "n_jobs": -1
    }
    base.update(params)
    base["monotone_constraints"] = mono
    return xgb.XGBClassifier(**base)

# 参数网格
PARAM_GRID = [
    {"max_depth":3, "min_child_weight":5.0, "reg_lambda":8.0, "learning_rate":0.05, "subsample":0.8},
    {"max_depth":3, "min_child_weight":6.0, "reg_lambda":10.0, "learning_rate":0.05, "subsample":0.8},
    {"max_depth":4, "min_child_weight":6.0, "reg_lambda":10.0, "learning_rate":0.04, "subsample":0.75},
    {"max_depth":3, "min_child_weight":8.0, "reg_lambda":12.0, "learning_rate":0.06, "subsample":0.8},
]

# 搜索参数并进行OOF校准
def search_params_and_oof_calibration(X, y, groups, mono, cap=0.10, n_splits=5):
    best_key = None
    best_score = (-1.0, -1.0)
    best_params = None
    best_oof = None
    pos = max(1, y.sum())
    neg = len(y) - pos
    base_spw = max(1.0, neg / pos)
    
    print(f"[交叉验证调试] 总样本数: {len(y)}, 阳性: {pos}, 阴性: {neg}, 基础权重: {base_spw:.2f}")
    
    for p_idx, p in enumerate(PARAM_GRID):
        params = dict(p)
        params.setdefault("scale_pos_weight", base_spw)
        oof_pred = np.zeros(len(y), dtype=float)
        fold_aucs = []
        
        for fold_idx, (tr_idx, val_idx) in enumerate(iter_group_folds(X, y, groups, n_splits=n_splits)):
            Xi, yi = X[tr_idx], y[tr_idx]
            Xv, yv = X[val_idx], y[val_idx]
            
            print(f"[交叉验证调试] 参数组合 {p_idx}, 折叠 {fold_idx}: 训练 {len(yi)}({yi.sum()}+), 验证 {len(yv)}({yv.sum()}+)")
            
            pos_i = max(1, yi.sum())
            neg_i = len(yi) - pos_i
            params["scale_pos_weight"] = max(1.0, neg_i/pos_i)
            clf = build_xgb(params, mono)
            clf.fit(Xi, yi, verbose=False)
            pred_fold = clf.predict_proba(Xv)[:,1]
            oof_pred[val_idx] = pred_fold
            
            if yv.sum() > 0:
                fold_auc = roc_auc_score(yv, pred_fold)
                fold_aucs.append(fold_auc)
                print(f"[交叉验证调试] 折叠 {fold_idx} AUC: {fold_auc:.4f}")
        
        if y.sum() > 0:
            rec_cap, thr_cap, fpr_cap = recall_at_fpr_cap(y, oof_pred, cap=cap)
            auc_ = roc_auc_score(y, oof_pred)
            ap_ = average_precision_score(y, oof_pred)
            print(f"[交叉验证调试] 参数组合 {p_idx}: OOF AUC={auc_:.4f}, PR-AUC={ap_:.4f}, 受限召回率={rec_cap:.4f}")
            key = (rec_cap, ap_)
            if key > best_score:
                best_score = key
                best_params = dict(p)
                best_oof = oof_pred
        else:
            print(f"[交叉验证调试] 参数组合 {p_idx}: 无阳性样本，跳过")
    
    if best_oof is not None and y.sum() > 0:
        oof_std = np.std(best_oof)
        oof_range = np.max(best_oof) - np.min(best_oof)
        print(f"[Platt校准调试] OOF预测 标准差={oof_std:.6f}, 范围={oof_range:.6f}")
        
        if oof_std > 1e-6 and oof_range > 1e-6:
            lr = LogisticRegression(solver="lbfgs", class_weight="balanced", random_state=Config.SEED, max_iter=1000)
            bz = np.clip(best_oof, 1e-6, 1-1e-6)
            z = np.log(bz/(1-bz)).reshape(-1,1)
            lr.fit(z, y.astype(int))
            print(f"[Platt校准调试] 校准成功，系数={lr.coef_[0][0]:.4f}, 截距={lr.intercept_[0]:.4f}")
        else:
            print(f"[Platt校准调试] OOF预测无变化，使用恒等校准")
            lr = IdentityCalibrator()
    else:
        print(f"[Platt校准调试] 无有效OOF，使用恒等校准")
        lr = IdentityCalibrator()
        best_params = PARAM_GRID[0]
        
    print(f"[交叉验证搜索] 选中参数: {best_params} | 最佳得分: {best_score}")
    return best_params, lr

# 训练最终XGBoost模型
def train_final_xgb(X_tr, y_tr, X_va, y_va, mono, params):
    clf = build_xgb(params, mono)
    clf.set_params(n_estimators=1000, early_stopping_rounds=100)
    clf.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_va, y_va)], verbose=False)
    return clf

# 选择阈值（带底线）
def pick_threshold_with_floor(y_val, p_val, X_val, fpr_cap=0.10, z_floor=1.5, fb=None):
    fpr, tpr, thr = roc_curve(y_val, p_val)
    qs = np.unique(np.quantile(p_val, np.linspace(0.01, 0.99, 33)))
    grid = sorted(set(list(thr) + list(qs)))
    best = (0.5, 0.0, 1.0)  # 阈值, 召回率, 假阳性率
    for t in grid:
        pred = (p_val >= t).astype(int)
        lowcov = ((X_val["unique_ratio"].values < (fb.lowcov_q_unique if fb else 0))
                  | (X_val["log_unique_reads"].values < (fb.lowcov_q_reads if fb else 0)))
        pred = np.where(lowcov & (X_val["absZmax"].values >= z_floor), 1, pred)
        tn, fp, fn, tp = confusion_matrix(y_val, pred).ravel()
        recall = tp / (tp + fn + 1e-12)
        fpr_ = fp / (fp + tn + 1e-12)
        if fpr_ <= fpr_cap and recall > best[1]:
            best = (t, recall, fpr_)
    return best

# 应用阈值
def apply_threshold(y_true, p, X, thr, z_floor, fb):
    pred = (p >= thr).astype(int)
    lowcov = ((X["unique_ratio"].values < fb.lowcov_q_unique) | (X["log_unique_reads"].values < fb.lowcov_q_reads))
    pred = np.where(lowcov & (X["absZmax"].values >= z_floor), 1, pred)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    prec = tp / (tp + fp + 1e-12)
    fpr_ = fp / (fp + tn + 1e-12)
    return {"acc": acc, "recall": recall, "precision": prec, "fpr": fpr_, "tp": tp, "fp": fp, "fn": fn, "tn": tn}, pred

# Platt变换
def platt_transform(lr, p):
    p = np.clip(np.asarray(p).ravel(), 1e-6, 1-1e-6)
    if hasattr(lr, 'coef_'):
        z = np.log(p / (1 - p)).reshape(-1, 1)
        return lr.predict_proba(z)[:, 1]
    else:
        return lr.predict_proba(p.reshape(-1, 1))[:, 1]

# 子类型召回率
def subtype_recall(df_part, y_true, pred_bin, cols=("Is_T13","Is_T18","Is_T21")):
    if not all(c in df_part.columns for c in cols): return None
    stats = {}
    for c in cols:
        idx = (df_part[c].values.astype(int) == 1)
        if idx.sum() > 0:
            tp = ((pred_bin[idx] == 1) & (y_true[idx] == 1)).sum()
            pos = int(y_true[idx].sum())
            stats[c] = {"pos": pos, "recall": float(tp/(pos+1e-12))}
    return stats

# 导出阈值曲线
def export_threshold_curve(y_true, p, path_csv):
    fpr, tpr, thr = roc_curve(y_true, p)
    pr_p, pr_r, pr_t = precision_recall_curve(y_true, p)
    n_pr = len(pr_p)
    n_thr = len(pr_t)
    if n_pr == n_thr + 1:
        pr_thresholds = list(pr_t) + [1.0]
    else:
        min_len = min(n_pr, n_thr)
        pr_thresholds = list(pr_t[:min_len])
        pr_p = pr_p[:min_len]
        pr_r = pr_r[:min_len]
    df1 = pd.DataFrame({"thr": thr, "fpr": fpr, "tpr(recall)": tpr})
    df2 = pd.DataFrame({
        "thr": pr_thresholds, 
        "precision": pr_p, 
        "recall": pr_r
    })
    df = df1.merge(df2, how="outer", on="thr", suffixes=('_roc', '_pr'))
    df.to_csv(path_csv, index=False, encoding="utf-8-sig")
    print(f"[导出] 阈值曲线已导出: {path_csv}")

# 导出可靠性分析
def export_reliability(y_true, p, bins=10, path_csv=None):
    df = pd.DataFrame({"y": y_true, "p": p})
    df = df.sort_values("p")
    df["bin"] = pd.qcut(df["p"], q=bins, duplicates='drop')
    out = df.groupby("bin").agg(y_rate=("y","mean"), p_mean=("p","mean"), n=("y","size")).reset_index()
    if path_csv:
        out.to_csv(path_csv, index=False, encoding="utf-8-sig")
    return out

# 导出特征重要性
def export_feature_importance(clf, feature_names, out_csv):
    booster = clf.get_booster()
    gain = booster.get_score(importance_type="gain")
    imp = [{"feature": feature_names[i], "gain": float(gain.get(f"f{i}", 0.0))} for i in range(len(feature_names))]
    imp_df = pd.DataFrame(imp).sort_values("gain", ascending=False)
    imp_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

# 主流程
df_raw = load_prenatal_data(Config.DATA_PATH)
y = (df_raw['染色体的非整倍体'].astype(str).str.strip() != "健康").astype(int).values
groups = df_raw['孕妇代码'].astype(str).fillna("NA").values

tr_idx, va_idx, te_idx = stratified_group_split(groups, y, 0.2, 0.2, Config.SEED)
tr = df_raw.iloc[tr_idx].reset_index(drop=True)
va = df_raw.iloc[va_idx].reset_index(drop=True)
te = df_raw.iloc[te_idx].reset_index(drop=True)
y_tr, y_va, y_te = y[tr_idx], y[va_idx], y[te_idx]
print(f"[数据集] 训练:{len(tr)} 验证:{len(va)} 测试:{len(te)} | 阳性数 Train/Valid/Test = {y_tr.sum()}/{y_va.sum()}/{y_te.sum()}")

fb = FeatureBuilder(z_thr=3.0).fit(tr)
X_tr = fb.transform(tr)
X_va = fb.transform(va)
X_te = fb.transform(te)

best_params, platt_lr = search_params_and_oof_calibration(X_tr.values, y_tr, groups[tr_idx], fb.mono, cap=Config.FPR_CAP, n_splits=5)

pos = max(1, y_tr.sum())
neg = len(y_tr) - pos
best_params["scale_pos_weight"] = max(1.0, neg/pos)
clf = train_final_xgb(X_tr.values, y_tr, X_va.values, y_va, fb.mono, best_params)

p_tr_raw = clf.predict_proba(X_tr.values)[:, 1]
p_va_raw = clf.predict_proba(X_va.values)[:, 1]
p_te_raw = clf.predict_proba(X_te.values)[:, 1]

p_tr = platt_transform(platt_lr, p_tr_raw)
p_va = platt_transform(platt_lr, p_va_raw)
p_te = platt_transform(platt_lr, p_te_raw)

print("---- 未校准 ----")
pr_metrics(y_tr, p_tr_raw, "训练集-原始")
pr_metrics(y_va, p_va_raw, "验证集-原始")
pr_metrics(y_te, p_te_raw, "测试集-原始")
print("---- Platt 校准后 ----")
pr_metrics(y_tr, p_tr, "训练集")
pr_metrics(y_va, p_va, "验证集")
pr_metrics(y_te, p_te, "测试集")

thr, best_rec, best_fpr = pick_threshold_with_floor(y_va, p_va, X_va, fpr_cap=Config.FPR_CAP, z_floor=Config.Z_FLOOR_LOW, fb=fb)
print(f"[验证集|假阳性率<= {Config.FPR_CAP:.2f}] 最佳阈值={thr:.4f}  召回率={best_rec:.4f}  假阳性率={best_fpr:.4f}")
res_va, _ = apply_threshold(y_va, p_va, X_va, thr, Config.Z_FLOOR_LOW, fb)
res_te, pred_te = apply_threshold(y_te, p_te, X_te, thr, Config.Z_FLOOR_LOW, fb)
print(f"[验证集@阈值] 准确率={res_va['acc']:.4f} 召回率={res_va['recall']:.4f} 精确率={res_va['precision']:.4f} 假阳性率={res_va['fpr']:.4f} TP={res_va['tp']} FP={res_va['fp']} FN={res_va['fn']} TN={res_va['tn']}")
print(f"[测试集@阈值] 准确率={res_te['acc']:.4f} 召回率={res_te['recall']:.4f} 精确率={res_te['precision']:.4f} 假阳性率={res_te['fpr']:.4f} TP={res_te['tp']} FP={res_te['fp']} FN={res_te['fn']} TN={res_te['tn']}")

sub_va = subtype_recall(va, y_va, (p_va>=thr).astype(int))
sub_te = subtype_recall(te, y_te, (p_te>=thr).astype(int))
if sub_va: print("[验证集子类型召回率]", sub_va)
if sub_te: print("[测试集子类型召回率]", sub_te)

os.makedirs(Config.ART_DIR, exist_ok=True)
export_threshold_curve(y_va, p_va, os.path.join(Config.ART_DIR, "valid_threshold_curves.csv"))
export_reliability(y_va, p_va, bins=8, path_csv=os.path.join(Config.ART_DIR, "valid_reliability.csv"))
export_feature_importance(clf, fb.feature_names, os.path.join(Config.ART_DIR, "feature_importance.csv"))

out = te[["序号"]].copy()
out["y_true"] = y_te
out["p_calibrated"] = p_te
out["pred@thr"] = (p_te>=thr).astype(int)
out.to_csv(os.path.join(Config.ART_DIR, "test_predictions.csv"), index=False, encoding="utf-8-sig")

bundle = {
    "feature_builder": fb,
    "xgb_model": clf,
    "platt_oof": platt_lr,
    "threshold": float(thr),
    "z_floor_low": float(Config.Z_FLOOR_LOW),
    "feature_names": fb.feature_names,
    "monotone": fb.mono,
    "best_params": best_params,
    "meta": {
        "seed": Config.SEED,
        "fpr_cap": float(Config.FPR_CAP),
        "data_path": Config.DATA_PATH,
        "artifact_dir": Config.ART_DIR
    }
}
with open(Config.PIPELINE_PATH, "wb") as f:
    pickle.dump(bundle, f)

print(f"已保存: {Config.PIPELINE_PATH}")