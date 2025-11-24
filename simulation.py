import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
import matplotlib.gridspec as gridspec
from adepy.uniform import point2
import os
from datetime import datetime

# 创建输出目录
output_dir = "single_source_scenarios"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class SingleSourceConfig:
    """单污染源实验配置基类"""

    def __init__(self):
        # 空间网格（统一设置）
        self.x = np.linspace(-10, 100, 200)
        self.y = np.linspace(-20, 20, 150)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        # 污染源参数（统一设置）
        self.c0 = 1000  # 源浓度 (mg/L)
        self.Q = 0.5  # 注入速率 (m³/d)
        self.xc, self.yc = 0, 0  # 点源位置


def create_scenario_plot(config, scenario_name, params_description):
    """创建标准化场景图"""
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2, wspace=0.3, hspace=0.4)

    times = [30, 90, 180, 365]  # 统一时间序列

    for i, t in enumerate(times):
        ax = fig.add_subplot(gs[i])

        # 计算浓度场
        C = point2(config.c0, config.X, config.Y, t, config.v, config.n,
                   config.al, config.ah, config.Q, config.xc, config.yc,
                   Dm=getattr(config, 'Dm', 0.0),
                   lamb=getattr(config, 'lamb', 0.0),
                   R=getattr(config, 'R', 1.0))

        # 绘制浓度分布
        contour = ax.contourf(config.X, config.Y, C, levels=30, cmap='viridis')
        ax.contour(config.X, config.Y, C, levels=[1, 10, 50, 100], colors='white', linewidths=0.8)

        # 标记污染源
        ax.plot(config.xc, config.yc, 'ro', markersize=10, markeredgecolor='white',
                markeredgewidth=2, label='污染源')

        # 添加水流方向
        ax.quiver(15, 15, 1, 0, scale=5, color='blue', width=0.015, label='水流方向')

        ax.set_title(f'第 {t} 天', fontsize=12, fontweight='bold')
        ax.set_xlabel('距离 (m)')
        ax.set_ylabel('距离 (m)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        plt.colorbar(contour, ax=ax, shrink=0.8, label='浓度 (mg/L)')

    # 添加参数说明文本框
    param_text = f"参数设置:\n流速 v = {config.v} m/d\n纵向弥散度 αL = {config.al} m\n横向弥散度 αT = {config.ah} m"
    if hasattr(config, 'lamb') and config.lamb > 0:
        param_text += f"\n衰减系数 λ = {config.lamb} $d^{{-1}}$"
    if hasattr(config, 'R') and config.R != 1.0:
        param_text += f"\n阻滞系数 R = {config.R}"

    fig.text(0.02, 0.02, param_text, fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

    plt.suptitle(f'{scenario_name}\n{params_description}',
                 fontsize=16, fontweight='bold', y=0.95)

    # 保存图片
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"{scenario_name.replace(' ', '_')}_{timestamp}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    return filename


def scenario_standard_industrial():
    """场景1：标准工业点源泄漏"""
    print("运行场景1：标准工业点源泄漏...")

    class Config(SingleSourceConfig):
        def __init__(self):
            super().__init__()
            self.v = 0.1  # 中等流速
            self.n = 0.25  # 孔隙度
            self.al = 2.0  # 典型纵向弥散度
            self.ah = 0.5  # 典型横向弥散度

    config = Config()
    scenario_name = "场景1：标准工业点源泄漏"
    params_description = "中等流速、典型弥散度 - 一般工业废水泄漏场景"

    filename = create_scenario_plot(config, scenario_name, params_description)

    return {
        "场景名称": scenario_name,
        "污染源类型": "连续点源",
        "适用情况": "典型工业废水泄漏、一般含水层条件",
        "主要参数": f"v={config.v} m/d, αL={config.al} m, αT={config.ah} m",
        "输出文件": filename
    }


def scenario_high_velocity():
    """场景2：高速水流条件下的污染迁移"""
    print("运行场景2：高速水流条件...")

    class Config(SingleSourceConfig):
        def __init__(self):
            super().__init__()
            self.v = 0.5  # 高流速
            self.n = 0.25
            self.al = 2.0
            self.ah = 0.5

    config = Config()
    scenario_name = "场景2：高速水流污染迁移"
    params_description = "高流速条件 - 高渗透性含水层、陡峭水力梯度"

    filename = create_scenario_plot(config, scenario_name, params_description)

    return {
        "场景名称": scenario_name,
        "污染源类型": "连续点源",
        "适用情况": "高渗透性含水层、陡峭水力梯度区域",
        "主要参数": f"v={config.v} m/d (5倍基准), αL={config.al} m, αT={config.ah} m",
        "输出文件": filename
    }


def scenario_strong_dispersion():
    """场景3：强弥散条件下的污染羽扩展"""
    print("运行场景3：强弥散条件...")

    class Config(SingleSourceConfig):
        def __init__(self):
            super().__init__()
            self.v = 0.1
            self.n = 0.25
            self.al = 10.0  # 强纵向弥散
            self.ah = 2.5  # 强横向弥散

    config = Config()
    scenario_name = "场景3：强弥散污染羽扩展"
    params_description = "强弥散作用 - 非均质性强、裂隙发育含水层"

    filename = create_scenario_plot(config, scenario_name, params_description)

    return {
        "场景名称": scenario_name,
        "污染源类型": "连续点源",
        "适用情况": "非均质性强、裂隙发育的含水层",
        "主要参数": f"v={config.v} m/d, αL={config.al} m (5倍基准), αT={config.ah} m (5倍基准)",
        "输出文件": filename
    }


def scenario_decaying_contaminant():
    """场景4：衰减污染物的迁移衰减"""
    print("运行场景4：衰减污染物...")

    class Config(SingleSourceConfig):
        def __init__(self):
            super().__init__()
            self.v = 0.1
            self.n = 0.25
            self.al = 2.0
            self.ah = 0.5
            self.lamb = 0.01  # 衰减系数

    config = Config()
    scenario_name = "场景4：衰减污染物迁移"
    params_description = "包含生物降解 - 可生物降解污染物自然衰减"

    filename = create_scenario_plot(config, scenario_name, params_description)

    return {
        "场景名称": scenario_name,
        "污染源类型": "连续点源",
        "适用情况": "可生物降解污染物（石油烃、苯系物等）",
        "主要参数": f"v={config.v} m/d, αL={config.al} m, αT={config.ah} m, λ={config.lamb} d⁻^{{-1}}",
        "输出文件": filename
    }


def scenario_retarded_contaminant():
    """场景5：吸附性污染物的阻滞迁移"""
    print("运行场景5：吸附性污染物...")

    class Config(SingleSourceConfig):
        def __init__(self):
            super().__init__()
            self.v = 0.1
            self.n = 0.25
            self.al = 2.0
            self.ah = 0.5
            self.R = 2.0  # 阻滞系数

    config = Config()
    scenario_name = "场景5：吸附性污染物阻滞迁移"
    params_description = "吸附阻滞作用 - 重金属、有机氯等强吸附性污染物"

    filename = create_scenario_plot(config, scenario_name, params_description)

    return {
        "场景名称": scenario_name,
        "污染源类型": "连续点源",
        "适用情况": "重金属、有机氯等吸附性强的污染物",
        "主要参数": f"v={config.v} m/d, αL={config.al} m, αT={config.ah} m, R={config.R}",
        "输出文件": filename
    }


def comparative_analysis():
    """对比分析：不同参数对污染迁移的影响"""
    print("进行参数对比分析...")

    # 创建对比图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # 定义对比场景
    scenarios = [
        {"v": 0.1, "al": 2.0, "ah": 0.5, "title": "基准场景", "color": "viridis"},
        {"v": 0.5, "al": 2.0, "ah": 0.5, "title": "高流速", "color": "viridis"},
        {"v": 0.1, "al": 10.0, "ah": 2.5, "title": "强弥散", "color": "viridis"},
        {"v": 0.1, "al": 2.0, "ah": 0.5, "lamb": 0.01, "title": "有衰减", "color": "viridis"},
        {"v": 0.1, "al": 2.0, "ah": 0.5, "R": 2.0, "title": "有阻滞", "color": "viridis"},
    ]

    t = 180  # 统一对比时间

    for i, scenario in enumerate(scenarios):
        ax = axes[i]

        # 计算浓度场
        C = point2(1000, *np.meshgrid(np.linspace(-10, 100, 150),
                                      np.linspace(-20, 20, 100)), t,
                   scenario["v"], 0.25, scenario["al"], scenario["ah"], 0.5, 0, 0,
                   lamb=scenario.get("lamb", 0.0),
                   R=scenario.get("R", 1.0))

        contour = ax.contourf(np.linspace(-10, 100, 150),
                              np.linspace(-20, 20, 100), C,
                              levels=30, cmap=scenario["color"])
        ax.contour(np.linspace(-10, 100, 150),
                   np.linspace(-20, 20, 100), C,
                   levels=[1, 10, 50], colors='white', linewidths=0.5)

        ax.set_title(scenario["title"], fontsize=14, fontweight='bold')
        ax.set_xlabel('距离 (m)')
        ax.set_ylabel('距离 (m)')
        ax.set_aspect('equal')
        plt.colorbar(contour, ax=ax, shrink=0.8, label='浓度 (mg/L)')

    # 隐藏最后一个子图
    axes[-1].set_visible(False)

    plt.suptitle('不同参数条件下污染迁移对比分析 (t=180天)',
                 fontsize=16, fontweight='bold', y=0.95)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"comparative_analysis_{timestamp}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    return filename


def run_all_single_source_scenarios():
    """运行所有单污染源场景"""
    print("=" * 70)
    print("单污染源多参数地下水污染实验场景模拟")
    print("=" * 70)
    print(f"结果将保存到目录: {output_dir}")

    scenarios = {
        "场景1": scenario_standard_industrial,
        "场景2": scenario_high_velocity,
        "场景3": scenario_strong_dispersion,
        "场景4": scenario_decaying_contaminant,
        "场景5": scenario_retarded_contaminant
    }

    results = {}

    # 运行所有场景
    for name, scenario_func in scenarios.items():
        try:
            results[name] = scenario_func()
            print(f"✓ {name} 完成")
        except Exception as e:
            print(f"✗ {name} 失败: {e}")
            results[name] = {"错误": str(e)}

    # 运行对比分析
    try:
        comparative_file = comparative_analysis()
        results["对比分析"] = {"输出文件": comparative_file}
        print("✓ 对比分析完成")
    except Exception as e:
        print(f"✗ 对比分析失败: {e}")

    # 生成总结报告
    generate_summary_report(results)

    return results


def generate_summary_report(results):
    """生成实验总结报告"""
    report_file = os.path.join(output_dir, "single_source_scenarios_summary.txt")

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("单污染源多参数地下水污染实验场景总结报告\n")
        f.write("=" * 60 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("实验设计概述:\n")
        f.write("- 所有场景均基于单一点污染源\n")
        f.write("- 通过调整物理参数模拟不同污染迁移特征\n")
        f.write("- 每个场景针对特定的水文地质条件或污染物类型\n\n")

        f.write("场景详细信息:\n")
        f.write("-" * 80 + "\n")

        for scenario_key, result in results.items():
            if scenario_key == "对比分析":
                continue

            f.write(f"\n{result.get('场景名称', scenario_key)}:\n")
            f.write(f"  污染源类型: {result.get('污染源类型', 'N/A')}\n")
            f.write(f"  适用情况: {result.get('适用情况', 'N/A')}\n")
            f.write(f"  主要参数: {result.get('主要参数', 'N/A')}\n")
            f.write(f"  输出文件: {result.get('输出文件', 'N/A')}\n")

    print(f"\n详细总结报告已保存至: {report_file}")


if __name__ == "__main__":
    results = run_all_single_source_scenarios()

    print("\n" + "=" * 70)
    print("单污染源多参数实验场景模拟完成！")
    print("=" * 70)
    print("实验场景总结:")
    print("1. 标准工业点源泄漏 - 基准参考场景")
    print("2. 高速水流条件 - 模拟高渗透性含水层")
    print("3. 强弥散条件 - 模拟非均质性含水层")
    print("4. 衰减污染物 - 包含生物降解过程")
    print("5. 吸附性污染物 - 包含吸附阻滞效应")
    print(f"\n所有结果已保存至: {output_dir}")