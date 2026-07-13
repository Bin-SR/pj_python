# -*- coding: utf-8 -*-
# =============================================================================
# VLA 自主抓取系统 —— 主入口
#
# 用法:
#   python main.py demo         收集演示数据
#   python main.py train        训练 VLA 策略网络
#   python main.py run          运行自主抓取 (含可视化)
#   python main.py run --script 脚本化抓取 (无需训练)
#   python main.py test         运行测试
# =============================================================================

import argparse
import sys
import os

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vla.config import ensure_dirs


def cmd_demo(args):
    """收集演示数据。"""
    from vla.env import VLAEnv
    from vla.demonstration import DemonstrationCollector

    print("=" * 60)
    print("  收集演示数据")
    print("=" * 60)

    env = VLAEnv(render=not args.headless)
    collector = DemonstrationCollector(env, render=False)

    try:
        collector.collect_dataset(
            num_episodes=args.episodes,
            cube_x_range=tuple(args.cube_x_range),
            cube_y_range=tuple(args.cube_y_range),
            save=True,
            filename=args.output,
        )
        print("\n演示数据收集完成!")
    finally:
        env.close()


def cmd_train(args):
    """训练 VLA 策略网络。"""
    from vla.train import train_from_demos

    print("=" * 60)
    print("  训练 VLA 策略网络")
    print("=" * 60)

    ensure_dirs()

    trainer, history = train_from_demos(
        demo_file=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )

    print("\n训练完成!")
    print(f"最终 train_loss: {history['train_losses'][-1]:.6f}")
    if history["val_losses"]:
        print(f"最终 val_loss: {history['val_losses'][-1]:.6f}")


def cmd_run(args):
    """运行自主抓取。"""
    from vla.inference import run_grasp_demo

    print("=" * 60)
    print("  自主抓取演示")
    print("=" * 60)

    run_grasp_demo(
        instruction=args.instruction,
        use_model=not args.script,
        model_path=args.model,
    )


def cmd_test(args):
    """运行测试: 验证各模块是否正常工作。"""
    print("=" * 60)
    print("  模块测试")
    print("=" * 60)

    # 1. 测试配置
    print("\n[1/5] 测试配置模块...")
    from vla.config import (
        SCENE_PATH, N_ARM_JOINTS, ACTION_DIM, JOINT_RANGES,
        normalize_joints, denormalize_joints,
    )
    print(f"  场景路径: {SCENE_PATH}")
    print(f"  关节数: {N_ARM_JOINTS}, 动作维度: {ACTION_DIM}")

    import numpy as np
    raw = np.array([0.0, 0.3, 0.0, -1.5, 0.0, 2.0, -0.5], dtype=np.float32)
    normed = normalize_joints(raw)
    recovered = denormalize_joints(normed)
    print(f"  归一化测试: max_err={np.max(np.abs(raw - recovered)):.6f}")

    # 2. 测试视觉模块
    print("\n[2/5] 测试视觉模块...")
    from vla.vision import VisualEncoder, RedCubeDetector
    import torch
    encoder = VisualEncoder()
    dummy = torch.randn(1, 3, 128, 128)
    out = encoder(dummy)
    print(f"  VisualEncoder: {dummy.shape} -> {out.shape}, "
          f"参数量={sum(p.numel() for p in encoder.parameters()):,}")
    detector = RedCubeDetector()
    print(f"  RedCubeDetector: 已初始化")

    # 3. 测试语言模块
    print("\n[3/5] 测试语言模块...")
    from vla.language import create_default_tokenizer, SimpleTextEncoder
    tokenizer = create_default_tokenizer()
    ids = tokenizer.encode("grasp the red cube")
    print(f"  分词: 'grasp the red cube' -> {ids.tolist()[:8]}...")
    text_enc = SimpleTextEncoder()
    batch = tokenizer.encode_batch(["grasp the red cube"])
    feat = text_enc(batch)
    print(f"  文本编码: {batch.shape} -> {feat.shape}, "
          f"参数量={sum(p.numel() for p in text_enc.parameters()):,}")

    # 4. 测试策略网络
    print("\n[4/5] 测试策略网络...")
    from vla.policy import VLAPolicy
    policy = VLAPolicy()
    dummy_img = torch.randn(2, 3, 128, 128)
    dummy_txt = tokenizer.encode_batch(["grasp the red cube", "pick up the block"])
    dummy_prop = torch.randn(2, 9)
    action = policy(dummy_img, dummy_txt, dummy_prop)
    print(f"  VLAPolicy: img={dummy_img.shape}, txt={dummy_txt.shape}, "
          f"prop={dummy_prop.shape}")
    print(f"            -> action={action.shape}, "
          f"参数量={sum(p.numel() for p in policy.parameters()):,}")

    # 5. 测试环境
    print("\n[5/5] 测试 MuJoCo 环境...")
    try:
        from vla.env import VLAEnv
        env = VLAEnv(render=False)
        obs = env.reset()
        print(f"  环境初始化成功")
        print(f"  观测: 图像={obs['image'].shape}, "
              f"手臂={obs['arm_qpos'].shape}, 方块={obs['cube_pos']}")

        # 测试 IK
        from vla.controller import IKSolver
        ik = IKSolver(
            env.get_model(), env.get_data(),
            env.get_arm_qpos_adr(), env.get_arm_dof_adr(),
            env.get_hand_body_id(),
        )
        cube = obs["cube_pos"]
        target = cube + np.array([0, 0, 0.15])
        qpos = ik.solve(target)
        print(f"  IK: 目标={target}, 解={np.round(qpos, 3)}")
        env.close()
        print(f"  环境测试通过!")
    except Exception as e:
        print(f"  环境测试跳过 (可能无 GUI): {e}")

    print("\n" + "=" * 60)
    print("  所有模块测试完成!")
    print("=" * 60)


def main():
    # 创建一个参数解释器，来读取命令行参数
    parser = argparse.ArgumentParser(
        description="VLA 自主抓取系统 - Franka Emika Panda + MuJoCo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py demo                       收集演示数据
  python main.py train                      训练模型
  python main.py run                        端到端自主抓取
  python main.py run --script               脚本化抓取 (无需训练)
  python main.py test                       运行模块测试
        """,
    )
 
    # 创建子命令，如python main.py XXXX
    # XXXX就是子命令(args.command)，可以说train, demo, run, test
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # ---- demo ---- : 创建一个demo的子命令
    demo_parser = subparsers.add_parser("demo", help="收集演示数据")
    demo_parser.add_argument("--episodes",      type=int, default=50, help="演示轮数 (默认: 50)")
    demo_parser.add_argument("--cube-x-range",  type=float, nargs=2, default=[0.35, 0.65], help="方块 x 范围")
    demo_parser.add_argument("--cube-y-range",  type=float, nargs=2, default=[-0.2, 0.2], help="方块 y 范围")
    demo_parser.add_argument("--output",        type=str, default="demo_data.pkl", help="输出文件名")
    demo_parser.add_argument("--headless",      action="store_true", help="无头模式 (不显示可视化)")

    # ---- train ----
    train_parser = subparsers.add_parser("train", help="训练 VLA 策略网络")
    train_parser.add_argument("--data",         type=str, default="demo_data.pkl", help="演示数据文件")
    train_parser.add_argument("--epochs",       type=int, default=50, help="训练轮数 (默认: 50)")
    train_parser.add_argument("--batch-size",   type=int, default=32, help="批大小 (默认: 32)")
    train_parser.add_argument("--device",       type=str, default="cuda", help="训练设备 (默认: cuda)")

    # ---- run ----
    run_parser = subparsers.add_parser("run", help="运行自主抓取")
    run_parser.add_argument("--instruction",    type=str, default="grasp the red cube", help="语言指令")
    run_parser.add_argument("--script",         action="store_true", help="使用脚本化抓取 (无需训练模型)")
    run_parser.add_argument("--model",          type=str, default="vla_policy.pt", help="模型文件名")

    # ---- test ----
    test_parser = subparsers.add_parser("test", help="运行模块测试")
    
    # 读取命令行输入，保存到args对象
    args = parser.parse_args()

    if args.command == "demo":
        cmd_demo(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "test":
        cmd_test(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
