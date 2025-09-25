import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings
import os
warnings.filterwarnings('ignore')

# 配置参数类

class Config:
    """配置参数类"""
    # 文件路径
    DATA_PATH = 'data/q2/q2_data.xlsx'
    RESULT_DIR = 'result/q3'
    RESULT_FILE = 'age_tstar/age_tstar_results.xlsx'
    
    # 成本参数
    C_RETEST = 500
    DELTA_T = 3/7  # 时间间隔（周）
    
    # 权重参数
    W_DELAY = 1    # 延迟成本权重
    W_RETEST = 1   # 重测成本权重
    
    # BMI自适应参数
    BMI0 = 32.0           # 参照BMI
    K_RETEST = 0.08       # 重测成本对BMI的敏感度
    CENTER0 = 14.5        # Sigmoid平滑中心
    CENTER_SLOPE = 0.05   # 中心随BMI的平移系数
    
    # 年龄相关参数
    K_AGE = 0.06                # 年龄对重测成本的乘法斜率
    K_AGE_CAP = 0.35            # 年龄乘法缩放的饱和上限
    CENTER_SLOPE_AGE = 0.04     # 失败概率Sigmoid中心对年龄的平移系数
    
    # DRO年龄感知参数
    THETA_AGE_MEAN = 0.25      # 年龄均值提高 → ρ 增加的强度
    THETA_AGE_VAR  = 0.35      # 年龄方差增大 → ρ 增加的强度
    ALPHA_AGE_VAR  = 0.30      # 段内年龄异质性惩罚强度
    KAPPA_AMA      = 0.30      # AMA优先系数（提高组权重）
    
    # 延迟成本参数
    DELAY_T_START = 10.0
    DELAY_BASE_GROWTH_RATE = 0.25
    
    # GAM模型参数
    SPLINE_DF_GA = 6
    SPLINE_DF_BMI = 4
    
    # DP参数
    GA_MIN = 10.0
    GA_MAX = 25.0
    MIN_GROUP_SIZE = 10
    MIN_BMI_WIDTH = 0.5

# 数据处理类

class DataProcessor:
    """数据加载和预处理类"""
    
    def __init__(self, config):
        self.config = config
        self.df = None
        self.df_gam = None
        self.df_woman = None
        self.age_mean = None
        self.age_std = None
    
    def load_and_preprocess(self):
        """加载并预处理数据"""
        try:
            self.df = pd.read_excel(self.config.DATA_PATH)
            self._preprocess_data()
            return True
        except FileNotFoundError:
            print(f"错误：数据文件 '{self.config.DATA_PATH}' 未找到。")
            return False
    
    def _preprocess_data(self):
        """数据预处理"""
        self.df = self.df[self.df['Y染色体浓度'].notna()].copy()
        self.df['孕周数'] = self.df['检测孕周'].apply(self._parse_weeks)
        self.df_gam = self.df[['Y染色体浓度', '孕周数', '孕妇BMI', '年龄', '孕妇代码']].dropna().copy()
        self.df_gam['woman_id'] = self.df_gam['孕妇代码'].astype('category')
        self.df_woman = self.df_gam.groupby('孕妇代码').agg({
            '孕妇BMI': 'mean',
            '年龄': 'mean'
        }).reset_index().sort_values('孕妇BMI').reset_index(drop=True)
        self.age_mean = float(self.df_woman['年龄'].mean()) if len(self.df_woman) else 30.0
        self.age_std = float(self.df_woman['年龄'].std()) if len(self.df_woman) else 5.0
        print(f"数据加载完成: {len(self.df_woman)} 位孕妇数据")
        print(f"年龄统计: 均值={self.age_mean:.2f}, 标准差={self.age_std:.2f}")
    
    @staticmethod
    def _parse_weeks(week_str):
        """解析孕周字符串"""
        if pd.isna(week_str): 
            return np.nan
        week_str = str(week_str).strip().lower().replace('w', '').replace('周', '')
        if '+' in week_str:
            parts = week_str.split('+')
            return float(parts[0]) + float(parts[1]) / 7
        return float(week_str)

# GAM模型类

class GAMModel:
    """广义加性模型类"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.sd_residuals = None
    
    def fit(self, data):
        """拟合GAM模型"""
        print("正在拟合GAM模型...")
        formula = (f"Y染色体浓度 ~ bs(孕周数, df={self.config.SPLINE_DF_GA}, degree=3) + "
                  f"bs(孕妇BMI, df={self.config.SPLINE_DF_BMI}, degree=3)")
        try:
            md = smf.mixedlm(formula, data=data, groups=data["woman_id"])
            self.model = md.fit(method=["lbfgs"], maxiter=2000)
        except Exception as e:
            print(f"混合效应模型拟合失败，使用OLS: {e}")
            self.model = smf.ols(formula, data=data).fit()
        y_pred = self.model.predict(data)
        residuals = data['Y染色体浓度'] - y_pred
        self.sd_residuals = np.std(residuals)
        print(f"模型拟合完成，残差标准差: {self.sd_residuals:.6f}")
        return self.model
    
    def predict(self, ga_weeks, bmi_values):
        """模型预测"""
        pred_data = pd.DataFrame({
            '孕周数': np.asarray(ga_weeks).flatten(), 
            '孕妇BMI': np.asarray(bmi_values).flatten()
        })
        return self.model.predict(pred_data).values.reshape(np.asarray(ga_weeks).shape)

# 成本计算器类（步骤一：个体化风险建模）

class CostCalculator:
    """个体化风险成本计算器类"""
    
    def __init__(self, config, gam_model, age_mean, age_std):
        self.config = config
        self.gam_model = gam_model
        self.memo_cache = {}
        self.age_mean = age_mean
        self.age_std = age_std
        self.max_delay_cost = self._calculate_max_delay_cost()
        self.max_retest_cost = config.C_RETEST * 15
        print(f"归一化参数: 最大延迟成本={self.max_delay_cost:.2f}, 估算最大重测成本={self.max_retest_cost}")
    
    def _calculate_max_delay_cost(self):
        return self.delay_cost_personalized(25, 35, 40)
    
    def delay_cost_personalized(self, t, bmi, age):
        """延迟成本（引入年龄因子调节）"""
        if t <= self.config.DELAY_T_START:
            return 0.0
        bmi_factor = 1.0 + 0.02 * (bmi - 25)
        age_factor = 1.0 + 0.01 * (age - 30)
        growth_rate = self.config.DELAY_BASE_GROWTH_RATE * bmi_factor * age_factor
        return np.exp(growth_rate * (t - self.config.DELAY_T_START)) - 1
    
    def prob_under_threshold(self, t, bmi):
        """基于GAM预测失败概率"""
        try:
            mu = self.gam_model.predict([t], [bmi])[0]
            return norm.cdf((0.04 - mu) / self.gam_model.sd_residuals) if np.isfinite(mu) else 0.5
        except Exception:
            return 0.5
    
    def smooth_retest_probability(self, t, base_prob, center=None, width=1.0):
        if center is None:
            center = self.config.CENTER0
        transition = 1 / (1 + np.exp(-(t - center) / width))
        return base_prob * (1 - transition) + 0.01 * transition
    
    def expected_retest_cost(self, t, bmi, age):
        """重测成本（引入年龄敏感的失败概率与缩放）"""
        state = (round(t, 2), round(bmi, 2), int(age))
        if state in self.memo_cache:
            return self.memo_cache[state]
        if t > 24:
            return 0.0
        center = (self.config.CENTER0 + 
                  self.config.CENTER_SLOPE * (bmi - self.config.BMI0) + 
                  self.config.CENTER_SLOPE_AGE * (age - self.age_mean))
        base_fail = self.prob_under_threshold(t, bmi)
        p_fail = self.smooth_retest_probability(t, base_fail, center=center)
        next_t = t + self.config.DELTA_T
        future_cost = self.expected_retest_cost(next_t, bmi, age)
        cost_if_fail = (self.config.C_RETEST + 
                        self.delay_cost_personalized(next_t, bmi, age) + 
                        future_cost)
        lambda_bmi = np.exp(self.config.K_RETEST * (bmi - self.config.BMI0))
        raw_age = self.config.K_AGE * (age - self.age_mean)
        raw_age = max(-self.config.K_AGE_CAP, min(self.config.K_AGE_CAP, raw_age))
        lambda_age = np.exp(raw_age)
        result = p_fail * (lambda_bmi * lambda_age) * cost_if_fail
        self.memo_cache[state] = result
        return result
    
    def total_normalized_cost(self, t, bmi, age):
        """总个体损失 L(t, BMI, Age)"""
        delay_cost = self.delay_cost_personalized(t, bmi, age)
        retest_cost = self.expected_retest_cost(t, bmi, age)
        normalized_delay = self.config.W_DELAY * delay_cost / self.max_delay_cost
        normalized_retest = self.config.W_RETEST * retest_cost / self.max_retest_cost
        return normalized_delay + normalized_retest

# DRO优化器类（步骤二：构建DP成本矩阵）

class DROOptimizer:
    """分布鲁棒优化器类"""
    
    def __init__(self, config, age_mean, age_std):
        self.config = config
        self.ga_grid = np.arange(config.GA_MIN, config.GA_MAX + 0.1, config.DELTA_T)
        self.age_mean = age_mean
        self.age_std = age_std
    
    @staticmethod
    def log_mean_exp(x):
        return logsumexp(x) - np.log(len(x))
    
    def kl_dual_value(self, loss_slice, rho):
        def g(eta):
            if eta <= 1e-6:
                return np.inf
            v = self.log_mean_exp(eta * loss_slice)
            return (rho + v) / eta
        res = minimize_scalar(g, bounds=(1e-6, 100.0), method='bounded')
        return res.fun
    
    def segment_cost(self, i, j, c_rho, loss_matrix, total_women, df_woman):
        """计算段成本（年龄感知的鲁棒预算调节）"""
        n = j - i + 1
        if (n < self.config.MIN_GROUP_SIZE or 
            (df_woman['孕妇BMI'].iloc[j] - df_woman['孕妇BMI'].iloc[i] < self.config.MIN_BMI_WIDTH)):
            return np.inf, None
        ages = df_woman['年龄'].iloc[i:j+1].values
        age_mean = float(np.mean(ages))
        age_var = float(np.var(ages, ddof=1)) if n > 1 else 0.0
        rho = (c_rho * n / total_women) * (
            1.0 
            + self.config.THETA_AGE_MEAN * (age_mean - self.age_mean) / max(1.0, self.age_std)
            + self.config.THETA_AGE_VAR * (age_var / (self.age_std**2 + 1e-8))
        )
        robust_costs = []
        for k in range(len(self.ga_grid)):
            loss_slice = loss_matrix[i:j+1, k]
            robust_cost = self.kl_dual_value(np.array(loss_slice), rho)
            robust_costs.append(robust_cost)
        k_star = np.argmin(robust_costs)
        min_robust_cost = robust_costs[k_star]
        age_heterogeneity_penalty = self.config.ALPHA_AGE_VAR * (age_var / (self.age_std**2 + 1e-8))
        ama_weight = 1.0 + self.config.KAPPA_AMA * (1.0 / (1.0 + np.exp(-(age_mean - 35.0))))
        weighted_cost = (n / total_women) * ama_weight * (min_robust_cost + age_heterogeneity_penalty)
        return weighted_cost, self.ga_grid[k_star]

# 动态规划求解器类（步骤三：动态规划求解全局最优分组）

class DPSolver:
    """动态规划求解器类"""
    
    @staticmethod
    def solve_optimal_segmentation(cost_matrix, num_groups):
        S = cost_matrix.shape[0]
        DP = np.full((S, num_groups + 1), np.inf)
        Prev = -np.ones((S, num_groups + 1), dtype=int)
        for s in range(S):
            DP[s, 1] = cost_matrix[0, s]
        for g in range(2, num_groups + 1):
            for s in range(g - 1, S):
                p_range = np.arange(g - 2, s)
                if p_range.size == 0:
                    continue
                costs = DP[p_range, g - 1] + cost_matrix[p_range + 1, s]
                min_idx = np.argmin(costs)
                best_p = p_range[min_idx]
                min_cost = costs[min_idx]
                if min_cost < np.inf:
                    DP[s, g] = min_cost
                    Prev[s, g] = best_p
        if DP[S - 1, num_groups] == np.inf:
            return [], np.inf
        cuts = []
        s_curr = S - 1
        for g_curr in reversed(range(2, num_groups + 1)):
            p = Prev[s_curr, g_curr]
            if p == -1:
                return [], np.inf
            cuts.append((p + 1, s_curr))
            s_curr = p
        cuts.append((0, s_curr))
        return list(reversed(cuts)), DP[S - 1, num_groups]

# 个体优化器类（步骤五：分组内个体最优时间计算）

class IndividualOptimizer:
    """个体最优T*计算器"""
    
    def __init__(self, cost_calculator):
        self.cost_calculator = cost_calculator
        self.ga_min = 10.0
        self.ga_max = 25.0
    
    def find_optimal_T(self, bmi, age):
        self.cost_calculator.memo_cache.clear()
        def objective(t):
            return self.cost_calculator.total_normalized_cost(t, bmi, age)
        result = minimize_scalar(
            objective,
            bounds=(self.ga_min, self.ga_max),
            method='bounded'
        )
        return result.x if result.success else 15.0

# 主优化器类（整合步骤一至七）

class PregnancyTestOptimizer:
    """孕妇检测优化主分析器"""
    
    def __init__(self):
        self.config = Config()
        self.data_processor = DataProcessor(self.config)
        self.gam_model = None
        self.cost_calculator = None
        self.dro_optimizer = None
        self.dp_solver = DPSolver()
        self.individual_optimizer = None
        self.df_woman = None
        self.results_df = None
        self.best_model = None
    
    def run_analysis(self, c_rho_candidates, g_candidates):
        """运行完整分析流程（步骤一至四）"""
        if not self.data_processor.load_and_preprocess():
            return None, None
        self.gam_model = GAMModel(self.config)
        self.gam_model.fit(self.data_processor.df_gam)
        self.cost_calculator = CostCalculator(
            self.config, self.gam_model, 
            self.data_processor.age_mean, self.data_processor.age_std
        )
        self.dro_optimizer = DROOptimizer(
            self.config, self.data_processor.age_mean, self.data_processor.age_std
        )
        loss_matrix = self._build_loss_matrix()
        self.results_df = self._run_dp_dro_analysis(c_rho_candidates, g_candidates, loss_matrix)
        if self.results_df is None:
            return None, None
        self.df_woman = self.data_processor.df_woman
        self._select_best_model()  # 步骤四
        return self.results_df, self.df_woman
    
    def _build_loss_matrix(self):
        """构建损失矩阵（步骤二部分）"""
        df_woman = self.df_woman
        ga_grid = self.dro_optimizer.ga_grid
        loss_matrix = np.zeros((len(df_woman), len(ga_grid)))
        print("正在构建损失矩阵...")
        for s, woman in tqdm(df_woman.iterrows(), total=len(df_woman)):
            self.cost_calculator.memo_cache.clear()
            for k, t in enumerate(ga_grid):
                cost = self.cost_calculator.total_normalized_cost(
                    t, woman['孕妇BMI'], woman['年龄']
                )
                loss_matrix[s, k] = cost
        return loss_matrix
    
    def _run_dp_dro_analysis(self, c_rho_candidates, g_candidates, loss_matrix):
        """运行DP+DRO分析（步骤二和三）"""
        results = []
        df_woman = self.df_woman
        for c_rho in tqdm(c_rho_candidates, desc="Processing c_rho"):
            parallel_results = Parallel(n_jobs=-1)(
                delayed(self.dro_optimizer.segment_cost)(i, j, c_rho, loss_matrix, len(df_woman), df_woman)
                for i in range(len(df_woman)) for j in range(i, len(df_woman))
            )
            cost_matrix = np.full((len(df_woman), len(df_woman)), np.inf)
            t_star_matrix = np.full((len(df_woman), len(df_woman)), np.nan)
            idx = 0
            for i in range(len(df_woman)):
                for j in range(i, len(df_woman)):
                    cost_matrix[i, j] = parallel_results[idx][0]
                    t_star_matrix[i, j] = parallel_results[idx][1]
                    idx += 1
            for G in g_candidates:
                groups, total_cost = self.dp_solver.solve_optimal_segmentation(cost_matrix, G)
                if not groups:
                    continue
                t_per_group = [t_star_matrix[gi, gj] for gi, gj in groups]
                bmi_intervals = [(df_woman['孕妇BMI'].iloc[gi], df_woman['孕妇BMI'].iloc[gj]) for gi, gj in groups]
                group_sizes = [gj - gi + 1 for gi, gj in groups]
                bic = len(df_woman) * np.log(total_cost + 1e-9) + (G * 2) * np.log(len(df_woman))
                results.append({
                    "c_rho": c_rho, "G": G, "BMI_intervals": bmi_intervals,
                    "T_star": t_per_group, "group_sizes": group_sizes,
                    "total_robust_risk": total_cost, "BIC": bic, "groups": groups
                })
        return pd.DataFrame(results)
    
    def _select_best_model(self):
        """模型选择与方案输出（步骤四）"""
        self.best_model = self.results_df.loc[self.results_df['BIC'].idxmin()]
        print(f"\n=== 最优模型 (BIC最小) ===")
        print(f"c_rho: {self.best_model['c_rho']}, G: {self.best_model['G']}")
        print(f"Total Robust Risk: {self.best_model['total_robust_risk']:.4f}")
        print(f"BIC: {self.best_model['BIC']:.2f}")
        print("-" * 50)
        for i in range(self.best_model['G']):
            bmi_low, bmi_high = self.best_model['BMI_intervals'][i]
            t_star = self.best_model['T_star'][i]
            size = self.best_model['group_sizes'][i]
            group_start, group_end = self.best_model['groups'][i]
            group_ages = self.df_woman['年龄'].iloc[group_start:group_end+1]
            age_mean = group_ages.mean()
            age_std = group_ages.std()
            print(f"组 {i+1}: BMI [{bmi_low:.2f}, {bmi_high:.2f}], "
                  f"最优孕周 {t_star:.2f} 周, 人数: {size}")
            print(f"        年龄: 均值 {age_mean:.1f} ± {age_std:.1f}")
    
    def compute_individual_t_stars(self):
        """分组内个体最优时间计算（步骤五）"""
        self.individual_optimizer = IndividualOptimizer(self.cost_calculator)
        df_individual = self.df_woman.copy()
        df_individual['T_star_individual'] = 0.0
        df_individual['group_id'] = -1
        # 分配组ID
        for i, (start_idx, end_idx) in enumerate(self.best_model['groups']):
            df_individual.loc[start_idx:end_idx, 'group_id'] = i
        print("正在计算个体最优T*...")
        for idx, row in tqdm(df_individual.iterrows(), total=len(df_individual)):
            t_star = self.individual_optimizer.find_optimal_T(row['孕妇BMI'], row['年龄'])
            df_individual.loc[idx, 'T_star_individual'] = t_star
        return df_individual
    
    def perform_regression_analysis(self, df_individual):
        """个性化检测时间回归建模（步骤六）"""
        regression_results = []
        for group_id in range(self.best_model['G']):
            group_data = df_individual[df_individual['group_id'] == group_id]
            if len(group_data) < 2:
                continue
            # 过滤异常T*（假设[10.5, 11.5]为异常范围，根据实际调整）
            filtered_data = group_data[(group_data['T_star_individual'] < 10.5) | (group_data['T_star_individual'] > 11.5)]
            if len(filtered_data) < 2:
                continue
            X = filtered_data['年龄'].values.reshape(-1, 1)
            y = filtered_data['T_star_individual'].values
            reg = LinearRegression()
            reg.fit(X, y)
            r2 = r2_score(y, reg.predict(X))
            group_ages = group_data['年龄']
            age_mean = group_ages.mean()
            regression_results.append({
                'group_id': group_id,
                'slope': reg.coef_[0],
                'intercept': reg.intercept_,
                'r2': r2,
                'n_points': len(filtered_data),
                'age_mean': age_mean,
                'equation': f'T* = {reg.coef_[0]:.3f} × (年龄 - {age_mean:.1f}) + 基准时间'
            })
        return regression_results
    
    def generate_personalized_scheme(self, regression_results):
        """最终个性化方案生成（步骤七）"""
        print("\n=== 最终个性化方案 ===")
        for i in range(self.best_model['G']):
            bmi_low, bmi_high = self.best_model['BMI_intervals'][i]
            base_t = self.best_model['T_star'][i]
            reg_info = next((r for r in regression_results if r['group_id'] == i), None)
            if reg_info:
                slope = reg_info['slope']
                age_mean = reg_info['age_mean']
                print(f"组 {i+1}: BMI [{bmi_low:.2f}, {bmi_high:.2f}]")
                print(f"  基准检测时间: {base_t:.2f} 周")
                print(f"  年龄调节斜率: {slope:.3f}")
                print(f"  个性化公式: 个体最优时间 = {base_t:.2f} + {slope:.3f} × (个体年龄 - {age_mean:.1f})")
            else:
                print(f"组 {i+1}: BMI [{bmi_low:.2f}, {bmi_high:.2f}]")
                print(f"  基准检测时间: {base_t:.2f} 周 (无足够数据进行年龄调节)")

# 主程序

def main():
    optimizer = PregnancyTestOptimizer()
    c_rho_candidates = [0.5]
    g_candidates = [4]
    print("=== 开始DP+DRO分析（加入年龄敏感性） ===")
    results_df, df_woman = optimizer.run_analysis(c_rho_candidates, g_candidates)
    if results_df is None:
        print("分析失败，请检查数据文件。")
        return
    print("\n=== 分析结果摘要 ===")
    print(results_df[['c_rho', 'G', 'total_robust_risk', 'BIC']])
    df_individual = optimizer.compute_individual_t_stars()
    regression_results = optimizer.perform_regression_analysis(df_individual)
    optimizer.generate_personalized_scheme(regression_results)

if __name__ == "__main__":
    main()