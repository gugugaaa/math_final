import pandas as pd
import io

def load_prenatal_data(file_path):
    """
    加载并初步处理产前检测数据。
    """
    # 1. 显式定义所有列名
    # 这样做可以确保列名的统一和规范，便于后续代码调用。
    # 注意：原文件表头在 'X染色体的Z值' 后有两个空列，这里我们为其命名以便识别，后续再决定如何处理。
    column_headers = [
        "序号", "孕妇代码", "年龄", "身高", "体重", "末次月经", "IVF妊娠", 
        "检测日期", "检测抽血次数", "检测孕周", "孕妇BMI", "原始读段数", 
        "在参考基因组上比对的比例", "重复读段的比例", "唯一比对的读段数", "GC含量",
        "13号染色体的Z值", "18号染色体的Z值", "21号染色体的Z值", "X染色体的Z值",
        "Unnamed_21", "Unnamed_22", "X染色体浓度", "13号染色体的GC含量", 
        "18号染色体的GC含量", "21号染色体的GC含量", "被过滤掉读段数的比例",
        "染色体的非整倍体", "怀孕次数", "生产次数", "胎儿是否健康"
    ]

    # 2. 加载数据
    # 使用 pd.read_excel 读取Excel文件，header=0 指定第一行为表头
    # names 参数提供我们自定义的列名列表
    df = pd.read_excel(file_path, header=0, names=column_headers)

    # 3. 初步数据清理
    # 'Unnamed_21' 和 'Unnamed_22' 这两列在数据中是完全空的，没有意义，可以直接删除。
    # 'X染色体浓度' 这一列数据缺失较多，后续分析时需要特别注意。
    df = df.drop(columns=["Unnamed_21", "Unnamed_22"])

    # 4. 处理孕周数据
    # 将"13w+5"格式转换为数值（周数+天数/7）
    def parse_gestational_week(week_str):
        if pd.isna(week_str):
            return None
        week_str = str(week_str).strip()
        if 'w' in week_str:
            parts = week_str.split('w')
            weeks = int(parts[0])
            if '+' in parts[1] and parts[1].strip() != '':
                days = int(parts[1].replace('+', ''))
                return weeks + days / 7
            else:
                return weeks
        return float(week_str) if week_str.replace('.', '').isdigit() else None
    
    df['检测孕周'] = df['检测孕周'].apply(parse_gestational_week)

    # 5. 处理怀孕次数
    # 将≥3的值统一改为3，同时处理数据类型转换
    def process_pregnancy_count(x):
        if pd.isna(x):
            return x
        try:
            # 转换为整数
            count = int(x)
            return min(count, 3)
        except (ValueError, TypeError):
            # 如果无法转换，返回原值
            return x
    
    df['怀孕次数'] = df['怀孕次数'].apply(process_pregnancy_count)

    # 6. 处理胎儿健康状态数据
    # 填充染色体非整倍体缺失值为'健康'
    df['染色体的非整倍体'] = df['染色体的非整倍体'].fillna('健康')

    # 使用与 xly2.py 完全相同的逻辑创建各种异常标识
    df['Is_T21'] = df['染色体的非整倍体'].apply(lambda x: 1 if 'T21' in x else 0)
    df['Is_T18'] = df['染色体的非整倍体'].apply(lambda x: 1 if 'T18' in x else 0)  
    df['Is_T13'] = df['染色体的非整倍体'].apply(lambda x: 1 if 'T13' in x else 0)

    # 验证构建结果
    print(f"胎儿健康状态统计:")
    print(f"T21 异常样本数: {df['Is_T21'].sum()}, T18: {df['Is_T18'].sum()}, T13: {df['Is_T13'].sum()}")
    print(f"染色体非整倍体类型分布:")
    print(df['染色体的非整倍体'].value_counts())

    print("数据加载完成，以下是数据概览：")
    df.info()
    print("\n" + "="*50 + "\n")
    
    return df