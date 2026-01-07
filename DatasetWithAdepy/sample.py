import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats.qmc import LatinHypercube  # 拉丁超立方抽样
from scipy.spatial import KDTree  # 快速找最近邻点


def sample_sparse_data(
        input_csv_dir: str,  # 原始CSV文件所在目录
        output_sparse_dir: str,  # 稀疏数据保存目录
        sample_strategy: str = "lhs",  # 新增lhs策略
        sample_ratio: float = 0.01,  # 采样比例（1%）
        sample_num: int = None,  # 固定采样数量（优先级更高）
        min_concentration: float = 0.1  # 最小有效浓度（过滤0值）
):
    """
    新增拉丁超立方（LHS）采样策略，适配PINN稀疏数据需求
    """
    os.makedirs(output_sparse_dir, exist_ok=True)

    # 获取所有CSV文件并按时间排序
    csv_files = [f for f in os.listdir(input_csv_dir) if f.startswith("全局浓度_") and f.endswith(".csv")]
    csv_files.sort(key=lambda x: int(x.split("_")[1].replace("天.csv", "")))

    for csv_file in tqdm(csv_files, desc="处理CSV文件（LHS采样）"):
        # 1. 读取原始数据并过滤无效点
        csv_path = os.path.join(input_csv_dir, csv_file)
        df = pd.read_csv(csv_path, encoding="utf-8")
        df_valid = df[df["污染物浓度_mg/L"] >= min_concentration].reset_index(drop=True)

        if len(df_valid) == 0:
            print(f"警告：{csv_file}无有效浓度点，跳过")
            continue

        # 2. 确定采样数量
        if sample_num is not None:
            n_sample = min(sample_num, len(df_valid))
        else:
            n_sample = max(1, int(len(df_valid) * sample_ratio))

        # 3. 采样策略（核心：新增LHS采样）
        if sample_strategy == "lhs":
            # 步骤1：获取X/Y坐标的范围（LHS采样的空间边界）
            x_min, x_max = df_valid["X坐标_m"].min(), df_valid["X坐标_m"].max()
            y_min, y_max = df_valid["Y坐标_m"].min(), df_valid["Y坐标_m"].max()

            # 步骤2：生成LHS采样点（2维：X和Y）
            lhs_sampler = LatinHypercube(d=2, seed=42)  # d=2表示X/Y二维，seed固定可复现
            lhs_sample = lhs_sampler.random(n=n_sample)  # 生成[0,1)范围内的采样点

            # 步骤3：将LHS采样点映射到实际X/Y坐标范围
            lhs_sample[:, 0] = x_min + lhs_sample[:, 0] * (x_max - x_min)  # X坐标映射
            lhs_sample[:, 1] = y_min + lhs_sample[:, 1] * (y_max - y_min)  # Y坐标映射

            # 步骤4：找到LHS采样点对应的原始数据中最近的有效点（KDTree加速）
            valid_coords = df_valid[["X坐标_m", "Y坐标_m"]].values
            kd_tree = KDTree(valid_coords)
            distances, indices = kd_tree.query(lhs_sample)  # 找最近邻索引

            # 步骤5：去重（避免多个LHS点匹配到同一个原始点）
            unique_indices = np.unique(indices)
            # 若去重后数量不足，补充随机采样
            if len(unique_indices) < n_sample:
                remaining = n_sample - len(unique_indices)
                supplement_indices = np.random.choice(
                    [i for i in range(len(df_valid)) if i not in unique_indices],
                    size=remaining, replace=False
                )
                final_indices = np.concatenate([unique_indices, supplement_indices])
            else:
                final_indices = unique_indices[:n_sample]

            # 步骤6：提取采样结果
            df_sample = df_valid.iloc[final_indices].reset_index(drop=True)

        # 兼容原有策略（random/uniform）
        elif sample_strategy == "random":
            df_sample = df_valid.sample(n=n_sample, random_state=42)
        elif sample_strategy == "uniform":
            n_bins = int(np.sqrt(n_sample))
            df_valid["x_bin"] = pd.cut(df_valid["X坐标_m"], bins=n_bins, labels=False)
            df_valid["y_bin"] = pd.cut(df_valid["Y坐标_m"], bins=n_bins, labels=False)
            df_sample = []
            for xb in df_valid["x_bin"].unique():
                for yb in df_valid["y_bin"].unique():
                    bin_df = df_valid[(df_valid["x_bin"] == xb) & (df_valid["y_bin"] == yb)]
                    if len(bin_df) > 0:
                        df_sample.append(bin_df.sample(1))
                    if len(df_sample) >= n_sample:
                        break
                if len(df_sample) >= n_sample:
                    break
            df_sample = pd.concat(df_sample).reset_index(drop=True).drop(columns=["x_bin", "y_bin"])
        else:
            raise ValueError(f"不支持的策略：{sample_strategy}，可选lhs/random/uniform")

        # 4. 保存稀疏数据
        output_path = os.path.join(output_sparse_dir, f"稀疏观测_{csv_file.replace('全局浓度_', '')}")
        df_sample.to_csv(output_path, index=False, encoding="utf-8")

        # 打印采样信息
        print(f"\n{csv_file} LHS采样完成：")
        print(f"  原始有效点数：{len(df_valid)} | 采样点数：{len(df_sample)}")
        print(f"  采样空间范围：X[{x_min:.1f}, {x_max:.1f}]m，Y[{y_min:.1f}, {y_max:.1f}]m")


# ===================== 示例调用（LHS采样） =====================
if __name__ == "__main__":
    # 示例1：LHS采样，1%比例（极稀疏）
    sample_sparse_data(
        input_csv_dir="场景1_单源_低注入速率/csv_results",  # 替换为你的目录
        output_sparse_dir="场景1_单源_低注入速率/稀疏观测数据_LHS_1%",
        sample_strategy="lhs",  # 启用LHS采样
        sample_ratio=0.01,  # 1%稀疏比例
        min_concentration=0
    )

    # 示例2：LHS采样，固定50个点（中等稀疏）
    # sample_sparse_data(
    #     input_csv_dir="场景2_双源_差异化Qa/csv_results",
    #     output_sparse_dir="场景2_双源_差异化Qa/稀疏观测数据_LHS_50点",
    #     sample_strategy="lhs",
    #     sample_num=50,  # 固定50个点
    #     min_concentration=0.1
    # )