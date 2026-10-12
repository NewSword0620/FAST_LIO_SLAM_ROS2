#!/usr/bin/env python3
"""画 KITTI 格式轨迹（优化前 vs 优化后），不依赖 evo。

三个标签页:
  Tab 1: XY 平面轨迹
  Tab 2: x/y/z 位置随时间
  Tab 3: roll/pitch/yaw 姿态随时间

用法:
    python3 plot_trajectory.py                 # 默认比较 save_data 下两文件
    python3 plot_trajectory.py 文件1 [文件2..]  # 指定文件
"""

import math
import os
import sys
import tkinter as tk
from tkinter import ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


def load_poses(filepath):
    """读取轨迹文件，自动识别 TUM 或 KITTI 格式。

    TUM  (8 列):  时间 tx ty tz qx qy qz qw
    KITTI(12 列): r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz

    返回: times, xs, ys, zs, rolls, pitches, yaws
    """
    times, xs, ys, zs = [], [], [], []
    rolls, pitches, yaws = [], [], []
    with open(filepath) as f:
        for line in f:
            parts = line.split()
            n = len(parts)
            if n < 8:
                continue
            v = [float(x) for x in parts]

            if n >= 12:
                # ---- KITTI：3x4 矩阵，平移在 [3][7][11] ----
                r11, r12, r13 = v[0], v[1], v[2]
                r21, r22, r23 = v[4], v[5], v[6]
                r31, r32, r33 = v[8], v[9], v[10]
                tx, ty, tz = v[3], v[7], v[11]
                t = 0.0
            else:
                # ---- TUM：时间 + 平移 + 四元数 ----
                t, tx, ty, tz = v[0], v[1], v[2], v[3]
                qx, qy, qz, qw = v[4], v[5], v[6], v[7]
                # 四元数 -> 旋转矩阵
                r11 = 1 - 2 * (qy * qy + qz * qz)
                r12 = 2 * (qx * qy - qz * qw)
                r13 = 2 * (qx * qz + qy * qw)
                r21 = 2 * (qx * qy + qz * qw)
                r22 = 1 - 2 * (qx * qx + qz * qz)
                r23 = 2 * (qy * qz - qx * qw)
                r31 = 2 * (qx * qz - qy * qw)
                r32 = 2 * (qy * qz + qx * qw)
                r33 = 1 - 2 * (qx * qx + qy * qy)

            times.append(t)
            xs.append(tx)
            ys.append(ty)
            zs.append(tz)

            # 旋转矩阵 -> ZYX 欧拉角 (roll=x, pitch=y, yaw=z)
            roll = math.atan2(r32, r33)
            pitch = math.atan2(-r31, math.sqrt(r32 * r32 + r33 * r33))
            yaw = math.atan2(r21, r11)
            rolls.append(math.degrees(roll))
            pitches.append(math.degrees(pitch))
            yaws.append(math.degrees(yaw))
    return times, xs, ys, zs, rolls, pitches, yaws


def find_save_data():
    """定位 save_data 目录：优先脚本同级目录，其次工作区根目录。"""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "save_data"),
        os.path.abspath(os.path.join(here, "..", "..", "save_data")),
    ]
    for d in candidates:
        if os.path.isdir(d):
            return d
    return candidates[0]


def _markevery(n):
    """根据点数自动控制标记密度，避免几千帧画成黑团。"""
    return max(1, n // 200)


def _xaxis(times):
    """横轴：优先时间戳（相对首帧，秒），否则退回帧索引。"""
    if times and times[-1] > 0:
        t0 = times[0]
        return [t - t0 for t in times]
    return list(range(len(times)))


def make_fig_xy(data):
    fig = Figure(figsize=(9, 7))
    ax = fig.add_subplot(111)
    for fp, (times, xs, ys, zs, *_rest) in data.items():
        n = len(xs)
        ax.plot(xs, ys, marker="o", linestyle="-", linewidth=1.0,
                markersize=3, markevery=_markevery(n),
                label=f"{os.path.basename(fp)} ({n} pts)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Trajectory (XY)")
    ax.axis("equal")
    ax.grid(True, alpha=0.4)
    ax.legend()
    fig.tight_layout()
    return fig


def make_fig_xyz(data):
    """Position 标签页：x / y / z 三个子图（3行1列），共享 x 轴联动。"""
    fig = Figure(figsize=(7, 8))
    axes = fig.subplots(3, 1, sharex=True)  # 共享 x 轴：缩放/平移同步
    for fp, (times, xs, ys, zs, *_rest) in data.items():
        t = _xaxis(times)
        name = os.path.basename(fp)
        n = len(xs)
        me = _markevery(n)
        axes[0].plot(t, xs, marker="o", linestyle="-", linewidth=0.8,
                     markersize=2.5, markevery=me, label=name)
        axes[1].plot(t, ys, marker="o", linestyle="-", linewidth=0.8,
                     markersize=2.5, markevery=me, label=name)
        axes[2].plot(t, zs, marker="o", linestyle="-", linewidth=0.8,
                     markersize=2.5, markevery=me, label=name)
    for ax, axis_name in zip(axes, ["x", "y", "z"]):
        ax.set_xlabel("time [s]")
        ax.set_ylabel(f"{axis_name} [m]")
        ax.set_title(f"Position {axis_name}")
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def make_fig_rpy(data):
    """Attitude 标签页：roll / pitch / yaw 三个子图（3行1列），共享 x 轴联动。"""
    fig = Figure(figsize=(7, 8))
    axes = fig.subplots(3, 1, sharex=True)  # 共享 x 轴：缩放/平移同步
    for fp, (times, *rest) in data.items():
        # rest = (xs, ys, zs, rl, pt, yw)，取后三个姿态
        rl, pt, yw = rest[-3], rest[-2], rest[-1]
        t = _xaxis(times)
        name = os.path.basename(fp)
        n = len(rl)
        me = _markevery(n)
        axes[0].plot(t, rl, marker="o", linestyle="-", linewidth=0.8,
                     markersize=2.5, markevery=me, label=name)
        axes[1].plot(t, pt, marker="o", linestyle="-", linewidth=0.8,
                     markersize=2.5, markevery=me, label=name)
        axes[2].plot(t, yw, marker="o", linestyle="-", linewidth=0.8,
                     markersize=2.5, markevery=me, label=name)
    for ax, axis_name in zip(axes, ["roll", "pitch", "yaw"]):
        ax.set_xlabel("time [s]")
        ax.set_ylabel("angle [deg]")
        ax.set_title(f"Attitude {axis_name}")
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def main():
    save_dir = find_save_data()
    default_odom = os.path.join(save_dir, "odom_poses.txt")
    default_optim = os.path.join(save_dir, "optimized_poses.txt")

    files = sys.argv[1:]
    if not files:
        files = [default_odom, default_optim]

    data = {}
    for fp in files:
        try:
            times, xs, ys, zs, rl, pt, yw = load_poses(fp)
        except OSError as e:
            print(f"[error] 无法读取 {fp}: {e}")
            continue
        if not xs:
            print(f"[warn] {fp} 为空或格式不正确")
            continue
        data[fp] = (times, xs, ys, zs, rl, pt, yw)
        print(f"[info] {os.path.basename(fp)}: {len(xs)} 帧")

    if not data:
        print("没有可绘制的轨迹数据。")
        sys.exit(1)

    # ---- 主窗口 + 标签页 ----
    root = tk.Tk()
    root.title("Trajectory Viewer")
    # 自适应：窗口 = 屏幕的 90%，保证工具栏可见
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    root.geometry(f"{int(screen_w * 0.9)}x{int(screen_h * 0.9)}")
    root.minsize(600, 400)

    nb = ttk.Notebook(root)
    nb.pack(fill="both", expand=True)

    figures = [
        ("Trajectory (XY)", make_fig_xy),
        ("Position x/y/z", make_fig_xyz),
        ("Attitude r/p/y", make_fig_rpy),
    ]

    for name, make_fig in figures:
        tab = ttk.Frame(nb)
        canvas = FigureCanvasTkAgg(make_fig(data), master=tab)
        canvas.draw()
        # 工具栏强制放顶部（NavigationToolbar2Tk 默认底部，重新 pack 到顶部）
        toolbar = NavigationToolbar2Tk(canvas, tab)
        toolbar.update()
        toolbar.pack(side=tk.TOP, fill=tk.X)
        # 画布占剩余空间
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        nb.add(tab, text=name)

    root.mainloop()


if __name__ == "__main__":
    main()
