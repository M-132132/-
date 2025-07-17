import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib

# 全局样式设置
plt.style.use('seaborn-v0_8')
plt.rcParams['font.family'] = 'SimHei'
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']


def load_tensorboard_data(log_dir):
    """加载TensorBoard日志数据"""
    event_files = [f for f in os.listdir(log_dir) if f.startswith('events')]
    if not event_files:
        raise FileNotFoundError(f"未找到TensorBoard事件文件于: {log_dir}")

    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    return event_acc


def extract_scalar_data(event_acc, tag):
    """提取指定的标量数据"""
    try:
        events = event_acc.Scalars(tag)
        return {
            'steps': [e.step for e in events],
            'values': [e.value for e in events],
            'wall_time': [e.wall_time for e in events]
        }
    except KeyError:
        print(f"[警告] 未找到指标: {tag}")
        return None


def smooth_data(y, window_size=5):
    """数据平滑处理"""
    if window_size > len(y):
        window_size = len(y) // 2 if len(y) > 1 else 1
    box = np.ones(window_size) / window_size
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def plot_metrics(metrics_data, ref_vx=None, figsize=(16, 12)):
    """绘制训练指标曲线"""
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    # 1. 奖励曲线
    if 'Episode_1/Rewards' in metrics_data and 'Overall/Average_Rewards' in metrics_data:
        ax = axs[0, 0]
        epi_data = metrics_data['Episode_1/Rewards']
        avg_data = metrics_data['Overall/Average_Rewards']

        ax.plot(epi_data['steps'], smooth_data(epi_data['values']),
                label='每回合奖励', color=COLORS[0], alpha=0.7)
        ax.plot(avg_data['steps'], smooth_data(avg_data['values']),
                label='平均奖励', color=COLORS[1], linewidth=2)
        ax.set_xlabel('训练步数')
        ax.set_ylabel('奖励值')
        ax.set_title('训练奖励曲线')
        ax.legend()
        ax.grid(True, alpha=0.3)
    # 2. 回合长度
    if 'Overall/Episode_Iterations' in metrics_data:
        ax = axs[0, 1]
        iter_data = metrics_data['Overall/Episode_Iterations']

        ax.plot(iter_data['steps'], iter_data['values'],
                label='回合步数', color=COLORS[0], marker='o', markersize=3, linestyle='None')

        # 添加移动平均线
        window_size = max(5, len(iter_data['values']) // 10)
        if window_size > 1:
            avg_line = smooth_data(iter_data['values'], window_size)
            ax.plot(iter_data['steps'], avg_line,
                    label=f'{window_size}次移动平均', color=COLORS[1], linewidth=2)
        ax.set_xlabel('回合数')
        ax.set_ylabel('步数')
        ax.set_title('回合持续时间')
        ax.legend()
        ax.grid(True, alpha=0.3)
    # 3. 行驶距离
    if 'Overall/Episode_Distances' in metrics_data:
        ax = axs[1, 0]
        dist_data = metrics_data['Overall/Episode_Distances']

        ax.plot(dist_data['steps'], dist_data['values'],
                label='行驶距离', color=COLORS[2], alpha=0.7)
        ax.set_xlabel('回合数')
        ax.set_ylabel('距离 (m)')
        ax.set_title('单回合行驶距离')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description='TensorBoard训练可视化工具')
    parser.add_argument('--log_dir', type=str, required=True,
                        help='包含tensorboard_logs的实验目录路径')
    parser.add_argument('--ref_vx', type=float, default=16.67,
                        help='目标速度值 (m/s)，默认60km/h=16.67m/s')
    parser.add_argument('--output', type=str, default='training_results.png',
                        help='输出图片文件名')
    args = parser.parse_args()
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    # 检查目录结构
    tensorboard_dir = os.path.join(args.log_dir, 'tensorboard_logs')
    if not os.path.exists(tensorboard_dir):
        raise FileNotFoundError(f"TensorBoard日志目录不存在: {tensorboard_dir}")
    # 加载数据
    event_acc = load_tensorboard_data(tensorboard_dir)
    # 提取关键指标
    metrics = {
        'Episode_1/Rewards': extract_scalar_data(event_acc, 'Episode_1/Rewards'),
        'Overall/Average_Rewards': extract_scalar_data(event_acc, 'Overall/Average_Rewards'),
        'Overall/Episode_Iterations': extract_scalar_data(event_acc, 'Overall/Episode_Iterations'),
        'Overall/Episode_Distances': extract_scalar_data(event_acc, 'Overall/Episode_Distances')
    }
    # 过滤空数据
    metrics = {k: v for k, v in metrics.items() if v is not None}
    if not metrics:
        raise ValueError("未找到有效的训练指标数据")
    # 绘制图像
    fig = plot_metrics(metrics, args.ref_vx)
    # 保存结果
    output_path = os.path.join(args.log_dir, args.output)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"训练可视化结果已保存至: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()