#!/usr/bin/env python3
"""实时监控训练进度"""

import json
import matplotlib.pyplot as plt
from pathlib import Path
import time
import argparse


def monitor_training(exp_dir: str, refresh_interval: int = 30):
    """实时监控训练进度"""
    exp_path = Path(exp_dir)
    history_file = exp_path / 'training_history.json'

    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    while True:
        try:
            # 读取最新的训练历史
            with open(history_file, 'r') as f:
                history = json.load(f)

            # 清除子图
            for ax in axes.flat:
                ax.clear()

            # 绘制损失
            axes[0, 0].plot(history['train_loss'], label='Train Loss')
            if 'val_loss' in history:
                axes[0, 0].plot(history['val_loss'], label='Val Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)

            # 绘制准确率
            if 'train_metrics' in history:
                train_acc = [m.get('accuracy', 0) for m in history['train_metrics']]
                axes[0, 1].plot(train_acc, label='Train Acc')
            if 'val_metrics' in history:
                val_acc = [m.get('accuracy', 0) for m in history['val_metrics']]
                axes[0, 1].plot(val_acc, label='Val Acc')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].set_title('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True)

            # 绘制F1分数
            if 'train_metrics' in history:
                train_f1 = [m.get('f1', 0) for m in history['train_metrics']]
                axes[1, 0].plot(train_f1, label='Train F1')
            if 'val_metrics' in history:
                val_f1 = [m.get('f1', 0) for m in history['val_metrics']]
                axes[1, 0].plot(val_f1, label='Val F1')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('F1 Score')
            axes[1, 0].set_title('F1 Score')
            axes[1, 0].legend()
            axes[1, 0].grid(True)

            # 显示当前状态
            axes[1, 1].text(0.1, 0.9, f"当前轮数: {len(history['train_loss'])}",
                            transform=axes[1, 1].transAxes)
            if 'val_loss' in history and history['val_loss']:
                axes[1, 1].text(0.1, 0.8, f"最新验证损失: {history['val_loss'][-1]:.4f}",
                                transform=axes[1, 1].transAxes)
            axes[1, 1].set_xlim(0, 1)
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].axis('off')

            plt.suptitle(f'Training Progress - {exp_dir}')
            plt.pause(0.1)

            time.sleep(refresh_interval)

        except FileNotFoundError:
            print(f"等待训练开始...")
            time.sleep(5)
        except KeyboardInterrupt:
            print("监控停止")
            break

    plt.ioff()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='监控训练进度')
    parser.add_argument('--exp_dir', type=str, required=True,
                        help='实验目录')
    parser.add_argument('--interval', type=int, default=30,
                        help='刷新间隔（秒）')

    args = parser.parse_args()
    monitor_training(args.exp_dir, args.interval)