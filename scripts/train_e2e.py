#!/usr/bin/env python3
"""端到端训练脚本"""

import argparse
import torch
import wandb
from pathlib import Path
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config
from data import KGDataLoader
from models import MoEKGC
from training import Trainer
from evaluation import Evaluator, BaselineComparison
from utils import setup_logger, set_seed, plot_training_history
from torch_geometric.loader import DataLoader


def main():
    parser = argparse.ArgumentParser(description='MoE-KGC 端到端训练')

    # 基础参数
    parser.add_argument('--data_dir', type=str, required=True,
                        help='数据目录（包含train/val/test子目录）')
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--exp_name', type=str, default='moe_kgc_experiment',
                        help='实验名称')

    # 训练参数
    parser.add_argument('--task', type=str, default='link_prediction',
                        choices=['link_prediction', 'entity_classification', 'relation_extraction'],
                        help='训练任务')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')

    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备 (cuda/cpu)')
    parser.add_argument('--use_wandb', action='store_true',
                        help='使用Weights & Biases记录')
    parser.add_argument('--eval_only', action='store_true',
                        help='仅评估模式')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='从检查点恢复')

    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 加载配置
    config = get_config(args.config)

    # 更新配置
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.lr

    # 创建实验目录
    exp_dir = Path(f'experiments/{args.exp_name}')
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 设置日志
    logger = setup_logger('moe_kgc', str(exp_dir / 'logs'))
    logger.info(f"开始实验: {args.exp_name}")

    # 初始化wandb
    if args.use_wandb:
        wandb.init(
            project="moe-kgc-franka",
            name=args.exp_name,
            config=config.__dict__
        )

    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # ========== 数据加载 ==========
    logger.info("加载数据...")
    data_loader = KGDataLoader(config)

    # 创建数据集
    train_dataset = data_loader.create_dataset(Path(args.data_dir) / 'train')
    val_dataset = data_loader.create_dataset(Path(args.data_dir) / 'val')
    test_dataset = data_loader.create_dataset(Path(args.data_dir) / 'test')

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,  # 每次一个图/子图
        shuffle=True,
        num_workers=4,
        collate_fn=train_dataset.get_collate_fn()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4
    )

    logger.info(f"训练集大小: {len(train_dataset)}")
    logger.info(f"验证集大小: {len(val_dataset)}")
    logger.info(f"测试集大小: {len(test_dataset)}")

    # ========== 模型初始化 ==========
    logger.info("初始化模型...")
    model = MoEKGC(config)
    logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ========== 仅评估模式 ==========
    if args.eval_only:
        if args.checkpoint:
            logger.info(f"加载检查点: {args.checkpoint}")
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])

        evaluator = Evaluator(model, config, device)
        results = evaluator.evaluate(test_loader, task=args.task)

        logger.info("评估结果:")
        logger.info(results['report'])

        # 保存结果
        with open(exp_dir / 'test_results.json', 'w') as f:
            json.dump(results['metrics'], f, indent=2)

        # 可视化
        evaluator.visualize_results(
            results['metrics'],
            save_path=str(exp_dir / 'test_visualization.png')
        )

        # 基线对比
        logger.info("与基线模型对比...")
        comparison = BaselineComparison(model, config, device)
        comparison_df = comparison.compare_models(test_loader, task=args.task)
        comparison_df.to_csv(exp_dir / 'baseline_comparison.csv', index=False)

        comparison.visualize_comparison(
            comparison_df,
            save_path=str(exp_dir / 'comparison_visualization.png')
        )

        return

    # ========== 训练 ==========
    trainer = Trainer(model, config, device)

    # 加载检查点
    if args.checkpoint:
        logger.info(f"从检查点恢复: {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)

    # 训练模型
    logger.info("开始训练...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        task=args.task,
        save_dir=str(exp_dir / 'checkpoints')
    )

    # ========== 保存训练历史 ==========
    with open(exp_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # 可视化训练历史
    plot_training_history(
        history,
        title=f"{args.exp_name} Training History",
        save_path=str(exp_dir / 'training_history.png')
    )

    # ========== 最终评估 ==========
    logger.info("最终评估...")

    # 加载最佳模型
    best_checkpoint = exp_dir / 'checkpoints' / 'best_model.pt'
    checkpoint = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    evaluator = Evaluator(model, config, device)
    final_results = evaluator.evaluate(test_loader, task=args.task)

    logger.info("最终测试结果:")
    logger.info(final_results['report'])

    # 保存最终结果
    with open(exp_dir / 'final_results.json', 'w') as f:
        json.dump(final_results['metrics'], f, indent=2)

    # ========== 知识图谱可视化 ==========
    logger.info("生成知识图谱可视化...")

    # 使用测试集的一个批次生成可视化
    test_batch = next(iter(test_loader))
    test_batch = trainer._move_batch_to_device(test_batch)

    with torch.no_grad():
        model.eval()
        outputs = model(test_batch, task=args.task)

    # 创建可视化
    from utils.kg_visualization import KnowledgeGraphVisualizer
    visualizer = KnowledgeGraphVisualizer(config)

    # 这里需要从data_loader获取图结构
    kg_path = exp_dir / 'knowledge_graph_prediction.html'
    data_loader.visualize_knowledge_graph(str(kg_path))

    logger.info(f"实验完成！结果保存在: {exp_dir}")

    if args.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()