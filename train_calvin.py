import sys

# 1. 先导入并执行修复补丁（必须在导入 lerobot 训练模块之前）
from fix_lerobot_loading import fix_lerobot_loading
fix_lerobot_loading(verbose=True)

# 2. 导入 LeRobot 官方的训练主函数
from lerobot.scripts.train import main

if __name__ == "__main__":
    # 3. 将 CMD 传入的所有参数原封不动地交给官方 train 函数
    sys.exit(main())