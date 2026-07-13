import argparse

# 创建一个参数解释器，来读取命令行参数
parser = argparse.ArgumentParser(description="关于命令行参数的示例",
                                 # formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=
                                 """
                                    test1 
                                    test2
                                 """)
# 创建子命令， args.mode
subparsers = parser.add_subparsers(dest="mode")

# 子命令demo
demo_parser = subparsers.add_parser("demo")

# 给子命令加入参数
demo_parser.add_argument("--headless",  action="store_true")

# 读取命令行输入，保存到args对象
args = parser.parse_args()

parser.print_help()
if args.headless:
    print("run without viewer")
else:
    print("run with viewer")