import pandas as pd
import numpy as np
from scipy.stats import norm, pearsonr, spearmanr
import statsmodels.api as sm
import statsmodels.formula.api as smf

# =============================================================================
# 配置参数
# =============================================================================

class GAMConfig:
    """GAM模型配置参数类"""
    DATA_PATH = 'data/q2/q2_data.xlsx'
    SPLINE_DF_GA = 6
    SPLINE_DF_BMI = 4

# =============================================================================
# 数据处理类
# =============================================================================

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

# =============================================================================
# GAM模型类
# =============================================================================

class GAMModel:
    """广义加性模型类"""

    def __init__(self, config=None):
        self.config = config if config is not None else GAMConfig()
        self.model = None
        self.sd_residuals = None
        self.data_processor = DataProcessor(self.config)
        self.fitted_data = None
        self.ga_range = None
        self.bmi_range = None

    def load_data(self):
        """加载数据"""
        return self.data_processor.load_and_preprocess()

    def fit(self, data=None):
        """拟合GAM模型"""
        if data is None:
            if not self.load_data():
                raise ValueError("数据加载失败")
            data = self.data_processor.df_gam

        self.fitted_data = data

        # 存储训练数据的范围
        self.ga_range = (data['孕周数'].min(), data['孕周数'].max())
        self.bmi_range = (data['孕妇BMI'].min(), data['孕妇BMI'].max())

        print("=== 正在拟合GAM模型... ===")
        print(f"训练数据范围 - 孕周: {self.ga_range[0]:.2f}-{self.ga_range[1]:.2f}, BMI: {self.bmi_range[0]:.2f}-{self.bmi_range[1]:.2f}")

        formula = (f"Y染色体浓度 ~ bs(孕周数, df={self.config.SPLINE_DF_GA}, degree=3) + "
                  f"bs(孕妇BMI, df={self.config.SPLINE_DF_BMI}, degree=3)")

        try:
            md = smf.mixedlm(formula, data=data, groups=data["woman_id"])
            self.model = md.fit(method=["lbfgs"], maxiter=2000)
            print("✓ 混合效应模型拟合成功")
        except Exception as e:
            print(f"混合效应模型拟合失败: {e}。退化为普通最小二乘法(OLS)。")
            self.model = smf.ols(formula, data=data).fit()
            print("✓ OLS模型拟合成功")

        # 计算残差标准差
        y_pred = self.model.predict(data)
        residuals = data['Y染色体浓度'] - y_pred
        self.sd_residuals = np.std(residuals)

        print(f"模型残差标准差 (SD): {self.sd_residuals:.6f}")
        return self.model

    def _validate_and_clip_inputs(self, ga_weeks, bmi_values):
        """验证并裁剪输入数据到训练范围内"""
        if self.ga_range is None or self.bmi_range is None:
            raise ValueError("模型尚未拟合，无法获取数据范围")

        ga_array = np.asarray(ga_weeks)
        bmi_array = np.asarray(bmi_values)

        ga_min, ga_max = self.ga_range
        bmi_min, bmi_max = self.bmi_range

        ga_buffer = (ga_max - ga_min) * 0.01
        bmi_buffer = (bmi_max - bmi_min) * 0.01

        ga_clipped = np.clip(ga_array, ga_min + ga_buffer, ga_max - ga_buffer)
        bmi_clipped = np.clip(bmi_array, bmi_min + bmi_buffer, bmi_max - bmi_buffer)

        if not np.allclose(ga_array, ga_clipped) or not np.allclose(bmi_array, bmi_clipped):
            print(f"警告: 输入数据超出训练范围，已自动裁剪到安全范围内")
            print(f"孕周范围: {ga_min:.2f}-{ga_max:.2f}, BMI范围: {bmi_min:.2f}-{bmi_max:.2f}")

        return ga_clipped, bmi_clipped

    def predict(self, ga_weeks, bmi_values):
        """模型预测"""
        if self.model is None:
            raise ValueError("模型尚未拟合，请先调用fit()方法")

        ga_clipped, bmi_clipped = self._validate_and_clip_inputs(ga_weeks, bmi_values)

        pred_data = pd.DataFrame({
            '孕周数': ga_clipped.flatten(), 
            '孕妇BMI': bmi_clipped.flatten()
        })

        if self.fitted_data is not None:
            for col in ['孕周数', '孕妇BMI']:
                if col in self.fitted_data.columns:
                    pred_data[col] = pred_data[col].astype(self.fitted_data[col].dtype)

        try:
            predictions = self.model.predict(pred_data)
            return predictions.values.reshape(np.asarray(ga_weeks).shape)
        except Exception as e:
            print(f"预测过程中出现错误: {e}")
            return np.full(np.asarray(ga_weeks).shape, 0.05)

    def prob_under_threshold(self, ga_weeks, bmi_values, threshold=0.04):
        """计算低于阈值的概率"""
        if self.model is None:
            raise ValueError("模型尚未拟合，请先调用fit()方法")

        try:
            mu = self.predict(ga_weeks, bmi_values)
            return norm.cdf((threshold - mu) / self.sd_residuals)
        except Exception as e:
            print(f"概率计算过程中出现错误: {e}")
            return np.full(np.asarray(ga_weeks).shape, 0.5)

    def get_model_summary(self):
        """获取模型摘要"""
        if self.model is None:
            return "模型尚未拟合"
        return self.model.summary()

    def get_data_info(self):
        """获取数据信息"""
        if self.data_processor.df_gam is None:
            return "数据尚未加载"

        info = {
            "总观测数": len(self.data_processor.df_gam),
            "孕妇数量": self.data_processor.df_gam['孕妇代码'].nunique(),
            "孕周范围": (self.data_processor.df_gam['孕周数'].min(), 
                      self.data_processor.df_gam['孕周数'].max()),
            "BMI范围": (self.data_processor.df_gam['孕妇BMI'].min(), 
                      self.data_processor.df_gam['孕妇BMI'].max()),
            "Y染色体浓度范围": (self.data_processor.df_gam['Y染色体浓度'].min(), 
                           self.data_processor.df_gam['Y染色体浓度'].max())
        }
        return info

    def _calculate_correlations(self):
        """计算相关系数"""
        if self.fitted_data is None:
            return {}

        y_conc = self.fitted_data['Y染色体浓度']
        ga_weeks = self.fitted_data['孕周数']
        bmi = self.fitted_data['孕妇BMI']

        pearson_ga = pearsonr(ga_weeks, y_conc)
        spearman_ga = spearmanr(ga_weeks, y_conc)
        pearson_bmi = pearsonr(bmi, y_conc)
        spearman_bmi = spearmanr(bmi, y_conc)

        return {
            'ga_pearson': pearson_ga,
            'ga_spearman': spearman_ga,
            'bmi_pearson': pearson_bmi,
            'bmi_spearman': spearman_bmi
        }

# =============================================================================
# 主函数
# =============================================================================

def main():
    """主函数：演示GAM模型功能"""
    print("=== GAM模型功能演示 ===")

    gam_model = GAMModel()

    try:
        print("正在拟合GAM模型...")
        gam_model.fit()

        print("\n=== 数据信息 ===")
        data_info = gam_model.get_data_info()
        for key, value in data_info.items():
            if isinstance(value, tuple):
                print(f"{key}: {value[0]:.2f} - {value[1]:.2f}")
            else:
                print(f"{key}: {value}")

        print("\n=== 模型摘要 ===")
        print(gam_model.get_model_summary())

    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
