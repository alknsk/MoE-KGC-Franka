#!/usr/bin/env python3
"""端到端训练脚本"""
import os
import argparse
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4,5'  # 只使用1-5号GPU
import torch
# import wandb
import json
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5"
# 设置CUDA环境变量，优化内存使用并避开0号GPU
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256,expandable_segments:True"

from data.dataset import FrankaKGDataset
from config import get_config
from data import KGDataLoader
from models import MoEKGC
from training import Trainer
from evaluation import Evaluator, BaselineComparison
from utils import setup_logger, set_seed, plot_training_history
from torch_geometric.loader import DataLoader
from pathlib import Path
from data.batch_dataloader import MoEKGCBatchDataLoader

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
    parser.add_argument('--batch_size', type=int, default=8,
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
    parser.add_argument('--num_neighbors', type=int, nargs='+', default=[25, 10],
                        help='邻居采样数量 (例如: --num_neighbors 25 10)')
    parser.add_argument('--sampling_method', type=str, default='neighbor',
                        choices=['neighbor', 'cluster', 'random'],
                        help='子图采样方法')
    

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
    #if args.use_wandb:
    #    wandb.init(
    #        project="moe-kgc-franka",
    #        name=args.exp_name,
    #        config=config.__dict__
    #    )

    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # ========== 数据加载 ==========
    logger.info("加载数据...")
    
    # 检查是否有预处理的PyG数据
    train_pyg_path = Path(args.data_dir) / 'train' / 'pyg_data.pt'
    val_pyg_path = Path(args.data_dir) / 'val' / 'pyg_data.pt'
    test_pyg_path = Path(args.data_dir) / 'test' / 'pyg_data.pt'
    
    if train_pyg_path.exists():
        logger.info("加载预处理的PyG数据...")
        train_data = torch.load(train_pyg_path, weights_only=False)
        val_data = torch.load(val_pyg_path, weights_only=False) if val_pyg_path.exists() else None
        test_data = torch.load(test_pyg_path, weights_only=False) if test_pyg_path.exists() else None
    else:
        logger.info("创建PyG数据...")
        data_loader = KGDataLoader(config)
        
        # 创建数据集并转换为PyG格式
        train_dataset = data_loader.create_dataset(Path(args.data_dir) / 'train')
        train_data = train_dataset.to_pyg_data()
        torch.save(train_data, train_pyg_path)
        
        if (Path(args.data_dir) / 'val').exists():
            val_dataset = data_loader.create_dataset(Path(args.data_dir) / 'val')
            val_data = val_dataset.to_pyg_data()
            torch.save(val_data, val_pyg_path)
        else:
            val_data = None
        
        if (Path(args.data_dir) / 'test').exists():
            test_dataset = data_loader.create_dataset(Path(args.data_dir) / 'test')
            test_data = test_dataset.to_pyg_data()
            torch.save(test_data, test_pyg_path)
        else:
            test_data = None

    # 创建批处理数据加载器
    logger.info("创建mini-batch数据加载器...")

    # 添加批处理相关参数
    num_neighbors = args.num_neighbors
    sampling_method = args.sampling_method

    train_loader_manager = MoEKGCBatchDataLoader(
        train_data,
        batch_size=args.batch_size,
        num_neighbors=num_neighbors,
        sampling_method=sampling_method,
        num_workers=4,
        shuffle=True,
        mode='train',
        task=args.task
    )

    # 根据任务获取相应的加载器
    if args.task == 'link_prediction':
        train_loader = train_loader_manager.get_link_prediction_loader()
    else:
        train_loader = train_loader_manager.get_node_classification_loader()

    # 验证集和测试集的加载器
    if val_data is not None:
        val_loader_manager = MoEKGCBatchDataLoader(
            val_data,
            batch_size=args.batch_size * 2,  # 验证时使用更大批次
            num_neighbors=num_neighbors,
            sampling_method=sampling_method,
            num_workers=4,
            shuffle=False,
            mode='val',
            task=args.task
        )
        
        if args.task == 'link_prediction':
            val_loader = val_loader_manager.get_link_prediction_loader()
        else:
            val_loader = val_loader_manager.get_node_classification_loader()
    else:
        val_loader = None

    if test_data is not None:
        test_loader_manager = MoEKGCBatchDataLoader(
            test_data,
            batch_size=args.batch_size * 2,
            num_neighbors=num_neighbors,
            sampling_method=sampling_method,
            num_workers=4,
            shuffle=False,
            mode='test',
            task=args.task
        )
        
        if args.task == 'link_prediction':
            test_loader = test_loader_manager.get_link_prediction_loader()
        else:
            test_loader = test_loader_manager.get_node_classification_loader()
    else:
        test_loader = None

    logger.info(f"数据加载完成:")
    logger.info(f"  训练批次数: {len(train_loader)}")
    if val_loader:
        logger.info(f"  验证批次数: {len(val_loader)}")
    if test_loader:
        logger.info(f"  测试批次数: {len(test_loader)}")

    # ========== 模型初始化 ==========
    logger.info("初始化模型...")

    # 验证CUDA设备设置
    if torch.cuda.is_available():
        print(f"可用的CUDA设备数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"设备 {i}: {torch.cuda.get_device_name(i)}")

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

    #if args.use_wandb:
    #    wandb.finish()


if __name__ == '__main__':
    main()