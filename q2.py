import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import mixedlm
from joblib import Parallel, delayed
from tqdm import tqdm

# 配置参数

class Config:
    """配置参数类"""
    # 文件路径
    DATA_PATH = 'data/q2/q2_id_short.xlsx'
    
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
    
    def load_and_preprocess(self):
        """加载并预处理数据"""
        try:
            self.df = pd.read_excel(self.config.DATA_PATH)
            self._validate_data()
            self._preprocess_data()
            return True
        except FileNotFoundError:
            print(f"错误：数据文件 '{self.config.DATA_PATH}' 未找到。")
            return False
    
    def _validate_data(self):
        """验证数据完整性"""
        print(f"原始数据集大小: {len(self.df)} 行")
        print(f"唯一孕妇代码数量: {self.df['孕妇代码'].nunique()}")
        
        unique_codes = self.df['孕妇代码'].unique()
        expected_codes = [f'A{str(i).zfill(3)}' for i in range(1, 51)]
        
        if set(unique_codes) == set(expected_codes):
            print("✓ 数据集确实只包含 A001-A050")
        else:
            print("⚠ 数据集包含其他孕妇代码")
    
    def _preprocess_data(self):
        """数据预处理"""
        # 过滤缺失值
        self.df = self.df[self.df['Y染色体浓度'].notna()].copy()
        
        # 解析孕周
        self.df['孕周数'] = self.df['检测孕周'].apply(self._parse_weeks)
        
        # 准备GAM数据
        self.df_gam = self.df[['Y染色体浓度', '孕周数', '孕妇BMI', '年龄', '孕妇代码']].dropna().copy()
        self.df_gam['woman_id'] = self.df_gam['孕妇代码'].astype('category')
        
        # 准备孕妇级别数据
        self.df_woman = self.df_gam.groupby('孕妇代码').agg({
            '孕妇BMI': 'mean',
            '年龄': 'mean'
        }).reset_index().sort_values('孕妇BMI').reset_index(drop=True)
        
        print(f"预处理完成: {len(self.df_woman)} 位孕妇数据")
    
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
        print("=== 正在拟合GAM模型... ===")
        
        formula = (f"Y染色体浓度 ~ bs(孕周数, df={self.config.SPLINE_DF_GA}, degree=3) + "
                  f"bs(孕妇BMI, df={self.config.SPLINE_DF_BMI}, degree=3)")
        
        try:
            md = smf.mixedlm(formula, data=data, groups=data["woman_id"])
            self.model = md.fit(method=["lbfgs"], maxiter=2000)
        except Exception as e:
            print(f"混合效应模型拟合失败: {e}。退化为普通最小二乘法(OLS)。")
            self.model = smf.ols(formula, data=data).fit()
        
        # 计算残差标准差
        y_pred = self.model.predict(data)
        residuals = data['Y染色体浓度'] - y_pred
        self.sd_residuals = np.std(residuals)
        
        print(f"模型残差标准差 (SD): {self.sd_residuals:.6f}")
        return self.model
    
    def predict(self, ga_weeks, bmi_values):
        """模型预测"""
        pred_data = pd.DataFrame({
            '孕周数': np.asarray(ga_weeks).flatten(), 
            '孕妇BMI': np.asarray(bmi_values).flatten()
        })
        return self.model.predict(pred_data).values.reshape(np.asarray(ga_weeks).shape)

# 成本函数类

class CostCalculator:
    """成本计算器类"""
    
    def __init__(self, config, gam_model):
        self.config = config
        self.gam_model = gam_model
        self.memo_cache = {}
        
        # 计算归一化参数
        self.max_delay_cost = self._calculate_max_delay_cost()
        self.max_retest_cost = config.C_RETEST * 15
        
        print(f"归一化参数: 最大延迟成本={self.max_delay_cost:.2f}, 估算最大重测成本={self.max_retest_cost}")
    
    def _calculate_max_delay_cost(self):
        """计算最大延迟成本用于归一化"""
        return self.delay_cost_personalized(25, 35, 40)  # 假设最大值
    
    def delay_cost_personalized(self, t, bmi, age):
        """个性化延迟成本函数"""
        if t <= self.config.DELAY_T_START:
            return 0.0
        
        bmi_factor = 1.0 + 0.02 * (bmi - 25)
        age_factor = 1.0 + 0.01 * (age - 30)
        growth_rate = self.config.DELAY_BASE_GROWTH_RATE * bmi_factor * age_factor
        
        return np.exp(growth_rate * (t - self.config.DELAY_T_START)) - 1
    
    def prob_under_threshold(self, t, bmi):
        """计算低于阈值的概率"""
        try:
            mu = self.gam_model.predict([t], [bmi])[0]
            return norm.cdf((0.04 - mu) / self.gam_model.sd_residuals) if np.isfinite(mu) else 0.5
        except Exception:
            return 0.5
    
    def smooth_retest_probability(self, t, base_prob, center=None, width=1.0):
        """平滑重测概率函数"""
        if center is None:
            center = self.config.CENTER0
        transition = 1 / (1 + np.exp(-(t - center) / width))
        return base_prob * (1 - transition) + 0.01 * transition
    
    def expected_retest_cost(self, t, bmi, age):
        """期望重测成本（带缓存）"""
        state = (round(t, 2), round(bmi, 2), age)
        if state in self.memo_cache:
            return self.memo_cache[state]
        
        if t > 24:
            return 0.0
        
        # 计算失败概率
        center = self.config.CENTER0 + self.config.CENTER_SLOPE * (bmi - self.config.BMI0)
        base_fail = self.prob_under_threshold(t, bmi)
        p_fail = self.smooth_retest_probability(t, base_fail, center=center)
        
        # 递归计算未来成本
        next_t = t + self.config.DELTA_T
        future_cost = self.expected_retest_cost(next_t, bmi, age)
        cost_if_fail = (self.config.C_RETEST + 
                       self.delay_cost_personalized(next_t, bmi, age) + 
                       future_cost)
        
        # BMI缩放
        lambda_bmi = np.exp(self.config.K_RETEST * (bmi - self.config.BMI0))
        result = p_fail * lambda_bmi * cost_if_fail
        
        self.memo_cache[state] = result
        return result
    
    def total_normalized_cost(self, t, bmi, age):
        """计算归一化的总成本"""
        delay_cost = self.delay_cost_personalized(t, bmi, age)
        retest_cost = self.expected_retest_cost(t, bmi, age)
        
        normalized_delay = self.config.W_DELAY * delay_cost / self.max_delay_cost
        normalized_retest = self.config.W_RETEST * retest_cost / self.max_retest_cost
        
        return normalized_delay + normalized_retest

# DRO优化器类

class DROOptimizer:
    """分布鲁棒优化器类"""
    
    def __init__(self, config):
        self.config = config
        self.ga_grid = np.arange(config.GA_MIN, config.GA_MAX + 0.1, config.DELTA_T)
    
    @staticmethod
    def log_mean_exp(x):
        """数值稳定的log-sum-exp"""
        return logsumexp(x) - np.log(len(x))
    
    def kl_dual_value(self, loss_slice, rho):
        """计算KL-DRO值"""
        def g(eta):
            if eta <= 1e-6:
                return np.inf
            v = self.log_mean_exp(eta * loss_slice)
            return (rho + v) / eta
        
        res = minimize_scalar(g, bounds=(1e-6, 100.0), method='bounded')
        return res.fun
    
    def segment_cost(self, i, j, c_rho, loss_matrix, total_women):
        """计算段成本"""
        n = j - i + 1
        
        # 约束条件检查
        if (n < self.config.MIN_GROUP_SIZE or 
            (j < len(loss_matrix) and i < len(loss_matrix) and 
             loss_matrix.shape[0] > max(i,j))):  # 简化BMI宽度检查
            return np.inf, None
        
        rho = c_rho * n / total_women
        
        # 计算每个时间点的鲁棒成本
        robust_costs = []
        for k in range(len(self.ga_grid)):
            loss_slice = loss_matrix[i:j+1, k]
            robust_cost = self.kl_dual_value(np.array(loss_slice), rho)
            robust_costs.append(robust_cost)
        
        k_star = np.argmin(robust_costs)
        min_robust_cost = robust_costs[k_star]
        weighted_cost = (n / total_women) * min_robust_cost
        
        return weighted_cost, self.ga_grid[k_star]

# 动态规划求解器类

class DPSolver:
    """动态规划求解器类"""
    
    @staticmethod
    def solve_optimal_segmentation(cost_matrix, num_groups):
        """求解最优分割问题"""
        S = cost_matrix.shape[0]
        DP = np.full((S, num_groups + 1), np.inf)
        Prev = -np.ones((S, num_groups + 1), dtype=int)
        
        # 初始化g=1的情况
        for s in range(S):
            DP[s, 1] = cost_matrix[0, s]
        
        # 动态规划主循环
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
        
        # 回溯最优路径
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

# 主分析器类

class PregnancyTestOptimizer:
    """孕妇检测优化主分析器"""
    
    def __init__(self):
        self.config = Config()
        self.data_processor = DataProcessor(self.config)
        self.gam_model = None
        self.cost_calculator = None
        self.dro_optimizer = DROOptimizer(self.config)
        self.dp_solver = DPSolver()
        
    def run_analysis(self, c_rho_candidates, g_candidates):
        """运行完整分析流程"""
        # 1. 数据加载和预处理
        if not self.data_processor.load_and_preprocess():
            return None, None
        
        # 2. 拟合GAM模型
        self.gam_model = GAMModel(self.config)
        self.gam_model.fit(self.data_processor.df_gam)
        
        # 3. 初始化成本计算器
        self.cost_calculator = CostCalculator(self.config, self.gam_model)
        
        # 4. 构建损失矩阵
        loss_matrix = self._build_loss_matrix()
        
        # 5. 运行DP+DRO分析
        results_df = self._run_dp_dro_analysis(c_rho_candidates, g_candidates, loss_matrix)
        
        return results_df, self.data_processor.df_woman
    
    def _build_loss_matrix(self):
        """构建损失矩阵"""
        df_woman = self.data_processor.df_woman
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
        """运行DP+DRO分析"""
        results = []
        df_woman = self.data_processor.df_woman
        
        for c_rho in tqdm(c_rho_candidates, desc="Processing c_rho"):
            # 并行计算所有段成本
            parallel_results = Parallel(n_jobs=-1)(
                delayed(self.dro_optimizer.segment_cost)(i, j, c_rho, loss_matrix, len(df_woman))
                for i in range(len(df_woman)) for j in range(i, len(df_woman))
            )
            
            # 构建成本矩阵
            cost_matrix = np.full((len(df_woman), len(df_woman)), np.inf)
            t_star_matrix = np.full((len(df_woman), len(df_woman)), np.nan)
            
            idx = 0
            for i in range(len(df_woman)):
                for j in range(i, len(df_woman)):
                    cost_matrix[i, j] = parallel_results[idx][0]
                    t_star_matrix[i, j] = parallel_results[idx][1]
                    idx += 1
            
            # DP求解
            for G in g_candidates:
                groups, total_cost = self.dp_solver.solve_optimal_segmentation(cost_matrix, G)
                if not groups:
                    continue
                
                # 提取结果信息
                t_per_group = [t_star_matrix[gi, gj] for gi, gj in groups]
                bmi_intervals = [(df_woman['孕妇BMI'][gi], df_woman['孕妇BMI'][gj]) for gi, gj in groups]
                group_sizes = [gj - gi + 1 for gi, gj in groups]
                
                bic = len(df_woman) * np.log(total_cost + 1e-9) + (G * 2) * np.log(len(df_woman))
                
                results.append({
                    "c_rho": c_rho, "G": G, "BMI_intervals": bmi_intervals,
                    "T_star": t_per_group, "group_sizes": group_sizes,
                    "total_robust_risk": total_cost, "BIC": bic, "groups": groups
                })
        
        return pd.DataFrame(results)

# 主程序

def main():
    """主程序入口"""
    # 初始化优化器
    optimizer = PregnancyTestOptimizer()
    
    # 设置参数
    c_rho_candidates = [1]
    g_candidates = [2]
    
    # 运行分析
    print("=== 开始DP+DRO分析 ===")
    results_df, df_woman = optimizer.run_analysis(c_rho_candidates, g_candidates)
    
    if results_df is None:
        print("分析失败，请检查数据文件。")
        return
    
    # 显示结果
    print("\n=== 分析结果摘要 ===")
    print(results_df[['c_rho', 'G', 'total_robust_risk', 'BIC']])
    
    # 找到最优模型
    best_model = results_df.loc[results_df['BIC'].idxmin()]
    
    print(f"\n=== 最优模型 (BIC最小) ===")
    print(f"c_rho: {best_model['c_rho']}, G: {best_model['G']}")
    print(f"Total Robust Risk: {best_model['total_robust_risk']:.4f}")
    print(f"BIC: {best_model['BIC']:.2f}")
    print("-" * 50)
    
    for i in range(best_model['G']):
        bmi_low, bmi_high = best_model['BMI_intervals'][i]
        t_star = best_model['T_star'][i]
        size = best_model['group_sizes'][i]
        print(f"组 {i+1}: BMI [{bmi_low:.2f}, {bmi_high:.2f}], "
              f"最优孕周 {t_star:.2f} 周, 人数: {size}")

if __name__ == "__main__":
    main()