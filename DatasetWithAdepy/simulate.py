import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from adepy.uniform import point2


def create_contamination_scenario(
        # 场景核心配置（必选）
        scene_name: str,
        # 水文参数（默认=场景1参数）
        Kαα: float = 0.8,
        Kββ: float = 0,
        θ: float = 0.22,
        αL: float = 40,
        αT: float = 8,
        # 模拟区域参数（默认=1300m×800m）
        x_range: tuple = (0, 1300),
        y_range: tuple = (0, 800),
        grid_num: int = 100,
        # 多污染源参数（核心：支持差异化浓度+差异化注入速率）
        sources: list = [(650, 400)],  # 格式：[(x1,y1), (x2,y2), ...]
        c0_list: list = [100],  # 各污染源初始浓度列表（mg/L）
        Qa_list: list = [1 / 10],  # 各污染源注入速率列表（m²/d）
        # 时间参数（默认=1年周期，指定观测时间点）
        stress_cycle_d: int = 365,
        obs_times: list = [0, 1, 10, 30, 60, 180, 300, 330]
):
    """
    生成地下水单物种多污染源扩散模拟场景的核心函数
    （相同污染物，不同污染源可设置不同初始浓度、不同注入速率，总浓度=各污染源浓度线性叠加）
    :param scene_name: 场景名称（如"场景1"）
    :param Kαα: α方向导水率 (m/d)
    :param Kββ: β方向导水率 (m/d)
    :param θ: 有效孔隙度（无量纲）
    :param αL: 纵向扩散率 (m)
    :param αT: 横向扩散率 (m)
    :param x_range: X轴模拟范围 (min, max)，单位m
    :param y_range: Y轴模拟范围 (min, max)，单位m
    :param grid_num: 网格点数（X/Y轴一致）
    :param sources: 污染源坐标列表，格式[(x1,y1), (x2,y2), ...]
    :param c0_list: 各污染源初始浓度列表 (mg/L)，长度必须与sources一致
    :param Qa_list: 各污染源注入速率列表 (m²/d)，长度必须与sources一致
    :param stress_cycle_d: 应力周期/模拟总时长 (天)
    :param obs_times: 观测时间点列表 (天)
    """
    # ===================== 关键：参数校验（避免数量不匹配） =====================
    if len(c0_list) != len(sources):
        raise ValueError(f"浓度列表长度（{len(c0_list)}）与污染源数量（{len(sources)}）不一致！")
    if len(Qa_list) != len(sources):
        raise ValueError(f"注入速率列表长度（{len(Qa_list)}）与污染源数量（{len(sources)}）不一致！")

    # ===================== 1. 场景文件夹创建 =====================
    SCENE_ROOT = scene_name
    CSV_DIR = os.path.join(SCENE_ROOT, "csv_results")
    PLOT_DIR = os.path.join(SCENE_ROOT, "plot_results")
    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    # ===================== 2. 场景参数记录（含差异化注入速率） =====================
    param_txt_path = os.path.join(SCENE_ROOT, f"{scene_name}_参数记录.txt")
    with open(param_txt_path, "w", encoding="utf-8") as f:
        f.write(f"========== {scene_name} 模拟参数 ==========\n")
        f.write(f"【水文参数】\n")
        f.write(f"α方向导水率Kαα (m/d)：{Kαα}\n")
        f.write(f"β方向导水率Kββ (m/d)：{Kββ}\n")
        f.write(f"有效孔隙度θ：{θ}\n")
        f.write(f"纵向扩散率αL (m)：{αL}\n")
        f.write(f"横向扩散率αT (m)：{αT}\n")
        f.write(f"孔隙流速v (m/d)：{Kαα / θ:.4f}（v=Kαα/θ）\n\n")

        f.write(f"【模拟区域参数】\n")
        f.write(f"X轴范围 (m)：{x_range[0]} ~ {x_range[1]}\n")
        f.write(f"Y轴范围 (m)：{y_range[0]} ~ {y_range[1]}\n")
        f.write(f"网格点数：{grid_num}\n\n")

        f.write(f"【污染源参数（多源+差异化浓度+差异化注入速率）】\n")
        f.write(f"污染源数量：{len(sources)}\n")
        for i, ((xc, yc), c0, qa) in enumerate(zip(sources, c0_list, Qa_list)):
            f.write(f"污染源{i + 1}：坐标({xc}, {yc}) m，初始浓度{c0} mg/L，注入速率{qa} m²/d\n")
        f.write(f"\n【时间参数】\n")
        f.write(f"应力周期/总时长 (天)：{stress_cycle_d}\n")
        f.write(f"观测时间点 (天)：{obs_times}\n")

    # ===================== 3. 基础配置初始化 =====================
    plt.switch_backend('Agg')
    plt.rcParams["font.family"] = "Microsoft YaHei"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300

    # 模拟网格生成
    x_grid_arr = np.linspace(x_range[0], x_range[1], grid_num)
    y_grid_arr = np.linspace(y_range[0], y_range[1], grid_num)
    x_grid, y_grid = np.meshgrid(x_grid_arr, y_grid_arr)

    # 孔隙流速计算
    v = Kαα / θ

    # ===================== 4. 预计算所有时间点浓度（多源差异化Qa+叠加） =====================
    all_concentrations = []
    for t in obs_times:
        if t == 0:
            total_conc = np.zeros_like(x_grid)
        else:
            total_conc = np.zeros_like(x_grid)
            # 核心：遍历每个污染源，取对应浓度+对应注入速率计算
            for (xc, yc), c0, qa in zip(sources, c0_list, Qa_list):
                single_source_conc = point2(
                    x=x_grid, y=y_grid, t=t,
                    v=v, al=αL, ah=αT, n=θ,
                    xc=xc, yc=yc, Qa=qa, c0=c0  # 每个源用独立的Qa和c0
                )
                total_conc += single_source_conc
        all_concentrations.append(total_conc)

    # 统一颜色刻度
    max_conc = np.max([np.max(conc) for conc in all_concentrations]) * 1.1
    unified_levels = np.linspace(0, max_conc, 50)

    # ===================== 5. 浓度计算+文件保存 =====================
    summary_data = []
    for idx, t in enumerate(obs_times):
        print(f"[{scene_name}] 正在处理时间点：{t}天")
        total_conc = all_concentrations[idx]

        # 保存CSV
        flat_x = x_grid.flatten()
        flat_y = y_grid.flatten()
        flat_conc = total_conc.flatten()
        global_df = pd.DataFrame({
            "X坐标_m": flat_x,
            "Y坐标_m": flat_y,
            "污染物浓度_mg/L": flat_conc
        })
        csv_path = os.path.join(CSV_DIR, f"全局浓度_{t}天.csv")
        global_df.to_csv(csv_path, index=False, encoding="utf-8")

        # 绘制浓度分布图（标注浓度+注入速率）
        fig, ax = plt.subplots(figsize=(12, 8))
        contour = ax.contourf(
            x_grid, y_grid, total_conc,
            levels=unified_levels, vmin=0, vmax=max_conc,
            cmap="jet", alpha=0.8
        )
        # 标注每个污染源的浓度+注入速率
        for i, ((xc, yc), c0, qa) in enumerate(zip(sources, c0_list, Qa_list)):
            ax.scatter(
                xc, yc, color="red", s=150, marker="*",
                edgecolor="darkred", linewidth=2,
                label=f"污染源{i + 1}（{c0}mg/L，{qa}m²/d）" if i == 0 else ""
            )
            ax.annotate(
                f"源{i + 1}：{c0}mg/L\nQa={qa}m²/d", (xc, yc),
                xytext=(10, 10), textcoords="offset points",
                fontsize=10, color="darkred", weight="bold"
            )
        # 图表样式
        ax.set_xlabel("X坐标 (m)", fontsize=12)
        ax.set_ylabel("Y坐标 (m)", fontsize=12)
        ax.set_title(f"{scene_name} - 污染物浓度分布（{t}天，{len(sources)}个污染源）", fontsize=14)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")
        cbar = plt.colorbar(contour, ax=ax, shrink=0.9)
        cbar.set_label("浓度 (mg/L)", fontsize=11)
        # 保存图片
        plot_path = os.path.join(PLOT_DIR, f"浓度分布图_{t}天.png")
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()

        # 记录汇总数据
        summary_data.append({
            "观测时间_d": t,
            "最大浓度_mg/L": np.max(flat_conc),
            "平均浓度_mg/L": np.mean(flat_conc),
            "浓度标准差": np.std(flat_conc)
        })

    # ===================== 6. 保存汇总时间序列图 =====================
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot([d["观测时间_d"] for d in summary_data], [d["最大浓度_mg/L"] for d in summary_data],
            color="#E63946", linewidth=2, marker="o", markersize=8, label="最大浓度")
    ax.plot([d["观测时间_d"] for d in summary_data], [d["平均浓度_mg/L"] for d in summary_data],
            color="#457B9D", linewidth=2, marker="s", markersize=8, label="平均浓度")
    # 标注数值
    for idx, row in enumerate(summary_data):
        ax.annotate(f"{row['最大浓度_mg/L']:.2f}", (row["观测时间_d"], row["最大浓度_mg/L"]),
                    xytext=(5, 5), textcoords="offset points", fontsize=9, color="#E63946")
        ax.annotate(f"{row['平均浓度_mg/L']:.2f}", (row["观测时间_d"], row["平均浓度_mg/L"]),
                    xytext=(5, -15), textcoords="offset points", fontsize=9, color="#457B9D")
    # 样式配置
    ax.set_xlabel("观测时间 (天)", fontsize=12)
    ax.set_ylabel("浓度 (mg/L)", fontsize=12)
    ax.set_title(f"{scene_name} - 浓度时间序列（{len(sources)}个污染源，应力周期={stress_cycle_d}天）", fontsize=14)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    # 保存汇总图
    summary_plot_path = os.path.join(PLOT_DIR, f"{scene_name}_浓度时间序列.png")
    plt.savefig(summary_plot_path, bbox_inches="tight")
    plt.close()

    # ===================== 7. 输出完成提示 =====================
    print(f"\n===== {scene_name} 生成完成 =====")
    print(f"场景根目录：{os.path.abspath(SCENE_ROOT)}")
    print(f"参数记录文件：{os.path.abspath(param_txt_path)}")
    print(f"CSV结果目录：{os.path.abspath(CSV_DIR)}")
    print(f"图片结果目录：{os.path.abspath(PLOT_DIR)}")
    print(f"统一颜色刻度范围：0 ~ {max_conc:.2f} mg/L")
    print(f"核心参数预览：Kαα={Kαα} m/d, 污染源数量={len(sources)}, 各源Qa={Qa_list} m²/d, 各源浓度={c0_list} mg/L")


# ===================== 示例调用（不同场景/不同注入速率） =====================
if __name__ == "__main__":
    # 示例1：单污染源（默认参数，仅调整注入速率）
    create_contamination_scenario(
        scene_name="场景1_单源_低注入速率",
        Qa_list=[1 / 20]  # 调低注入速率，延缓稳态
    )
    #
    # # 示例2：双污染源（不同浓度+不同注入速率）
    # create_contamination_scenario(
    #     scene_name="场景2_双源_差异化Qa",
    #     sources=[(400, 300), (800, 500)],
    #     c0_list=[100, 150],  # 源1：100mg/L，源2：150mg/L
    #     Qa_list=[1 / 10, 1 / 5],  # 源1：0.1m²/d，源2：0.2m²/d（泄漏更猛）
    #     Kαα=2.2,  # 降低流速，让各时间段变化更明显
    #     αL=100,  # 增大弥散度
    #     obs_times=[0, 1, 10, 30, 90, 180, 360, 720]
    # )


    # create_contamination_scenario(
    #     # 场景核心配置：明确标注文献关联
    #     scene_name="Paswan2023_胶体促进污染物迁移（含停滞区）",
    #     # 水文参数：匹配文献多孔介质与弥散特性
    #     Kαα=0.775,  # 砂质含水层导水率，中等水平，强化流动区集中效应
    #     Kββ=0,  # 单向流，横向无流动
    #     θ=0.3,  # 文献砂孔隙度0.299~0.31，取有效孔隙度0.3
    #     αL=0.3,  # 纵向扩散率，基于文献D和V计算（≈0.96 m），简化为0.3 m
    #     αT=0.03,  # 横向扩散率为纵向1/10，匹配文献低横向扩散
    #     # 模拟区域：野外尺度扩展，避免边界效应
    #     x_range=(0, 50),  # 纵向50 m，可观察完整plume迁移
    #     y_range=(0, 20),  # 横向20 m，匹配二维模拟但横向扩散弱的特征
    #     grid_num=200,  # 高密度网格，精准捕捉点源扩散
    #     # 多污染源：单源注入，匹配文献柱体实验
    #     sources=[(10, 10)],  # 上游中心位置，避免边界影响
    #     c0_list=[10],  # 污染物初始浓度10 mg/L，为DOM浓度（80 mg/L）的1/8，避免饱和吸附
    #     Qa_list=[0.001],  # 注入速率，匹配文献流量量级
    #     # 时间参数：覆盖短期突破到长期稳定
    #     stress_cycle_d=100,  # 总时长100天，完整模拟迁移过程
    #     obs_times=[0, 1, 5, 10, 20, 50, 80, 100]  # 关键时间点，验证文献中"早期突破、低尾迹"
    # )
    # create_contamination_scenario(
    #     # 场景名称：明确标注无DOM对照
    #     scene_name="Kan1990_无DOM组（对照）",
    #     # 水文参数：匹配单孔隙土壤，无DOM时污染物易滞留，弥散略高
    #     Kαα=0.35424,  # 砂质土壤导水率，与高DOM组一致（排除导水率干扰）
    #     Kββ=0,  # 单向柱实验，横向无流动
    #     θ=0.41,  # 实验孔隙度平均值（0.4~0.418），确保渗流速度与实验一致
    #     αL=0.1,  # 无DOM时污染物滞留增强，纵向扩散率略高于高DOM组（模拟轻微尾迹）
    #     αT=0.01,  # 横向扩散率为纵向1/10，与实验“横向扩散可忽略”一致
    #     # 模拟区域：与高DOM组相同，确保对比公平性
    #     x_range=(0, 30),  # 纵向30 m，覆盖野外扩展后的迁移范围
    #     y_range=(0, 10),  # 横向10 m，避免资源浪费
    #     grid_num=200,  # 高密度网格，精准捕捉“低迁移速度”的plume形态
    #     # 污染源参数：无DOM，污染物浓度与DOM组一致（排除浓度干扰）
    #     sources=[(5, 5)],  # 上游中心注入，与DOM组位置相同
    #     c0_list=[5],  # 污染物初始浓度5 mg/L（与高DOM组一致，仅DOM浓度差异）
    #     Qa_list=[0.0005],  # 注入速率匹配实验流量量级，与DOM组一致
    #     # 时间参数：覆盖无DOM组“晚突破、慢迁移”的时间尺度
    #     stress_cycle_d=60,  # 总时长60天，确保观察到无DOM组的完整迁移过程
    #     obs_times=[0, 0.5, 1, 3, 7, 15, 30, 60]  # 重点观测中期（7~30天）的滞留差异
    # )
    # create_contamination_scenario(
    #     # 场景名称：明确标注低DOM（BSA）浓度
    #     scene_name="Kan1990_低DOM组（BSA_250mg_L）",
    #     # 水文参数：低BSA浓度下，污染物滞留减弱，弥散低于无DOM组
    #     Kαα=0.35424,  # 与无DOM组、高DOM组一致，仅DOM浓度为变量
    #     Kββ=0,  # 单向流动，横向无扩散
    #     θ=0.41,  # 实验孔隙度，确保渗流速度统一
    #     αL=0.07,  # 纵向扩散率介于无DOM组（0.1）与高DOM组（0.05）之间，匹配中等迁移速度
    #     αT=0.007,  # 横向扩散率随纵向同步调整，保持1/10比例
    #     # 模拟区域：与其他组完全一致，保证对比有效性
    #     x_range=(0, 30),
    #     y_range=(0, 10),
    #     grid_num=200,
    #     # 污染源参数：低BSA浓度，体现“浓度梯度效应”
    #     sources=[(5, 5)],  # 相同注入位置，排除空间干扰
    #     c0_list=[5],  # 污染物浓度5 mg/L（与其他组一致）
    #     Qa_list=[0.0005],  # 注入速率不变，仅DOM浓度差异
    #     # 时间参数：重点观测“低DOM与无DOM的迁移差异”
    #     stress_cycle_d=60,
    #     obs_times=[0, 0.5, 1, 3, 7, 15, 30, 60]  # 7天左右可观察到低DOM组的迁移优势
    # )
    # create_contamination_scenario(
    #     # 场景名称：明确标注低DOM（Triton）浓度及化学特性
    #     scene_name="Kan1990_低DOM组（Triton_X_100_457mg_L）",
    #     # 水文参数：Triton对固相亲和力高，弥散略高于低BSA组
    #     Kαα=0.35424,  # 轻微降低导水率（模拟Triton吸附于固相导致的孔隙阻塞）
    #     Kββ=0,  # 单向流动，横向无扩散
    #     θ=0.41,  # 实验孔隙度，与其他组一致
    #     αL=0.08,  # 纵向扩散率介于无DOM组（0.1）与低BSA组（0.07）之间，匹配Triton的滞留特性
    #     αT=0.008,  # 横向扩散率随纵向调整，保持1/10比例
    #     # 模拟区域：与其他组一致，确保对比公平
    #     x_range=(0, 30),
    #     y_range=(0, 10),
    #     grid_num=200,
    #     # 污染源参数：低Triton浓度，体现“化学特性差异”
    #     sources=[(5, 5)],  # 相同注入位置
    #     c0_list=[5],  # 污染物浓度5 mg/L（统一变量）
    #     Qa_list=[0.0005],  # 注入速率不变
    #     # 时间参数：重点观测Triton与BSA的迁移差异
    #     stress_cycle_d=60,
    #     obs_times=[0, 0.5, 1, 3, 7, 15, 30, 60]  # 7~15天可观察到Triton组比BSA组滞留更明显
    # )
    #
    #
    # create_contamination_scenario(
    #     scene_name="Kan1990_高DOM组（Triton_X_100_1887mg_L）",
    #     # 水文参数：匹配单孔隙土壤与低弥散
    #     Kαα=0.35424,
    #     Kββ=0,
    #     θ=0.41,
    #     αL=0.05,
    #     αT=0.005,
    #     # 模拟区域：野外扩展，避免边界效应
    #     x_range=(0, 30),
    #     y_range=(0, 10),
    #     grid_num=200,
    #     # 污染源：上游中心注入，匹配柱实验
    #     sources=[(5, 5)],
    #     c0_list=[5],
    #     Qa_list=[0.0005],
    #     # 时间参数：覆盖短期突破与长期稳定
    #     stress_cycle_d=60,
    #     obs_times=[0, 0.5, 1, 3, 7, 15, 30, 60]
    # )
    # 示例3：三污染源（不同场景/不同Qa）
    # create_contamination_scenario(
    #     scene_name="场景3_三源_差异化Qa",
    #     sources=[(300, 200), (650, 400), (900, 600)],
    #     c0_list=[80, 120, 90],
    #     Qa_list=[1/15, 1/10, 1/8],  # 三个源注入速率依次增大
    #     x_range=(0, 3000), y_range=(0, 2000)  # 扩大模拟区域
    # )