# lerobot_scripts
深度学习与空间智能期末作业 数据集修复脚本、训练脚本、数据集版本降级脚本
数据集修复脚本：fix_dataset_stats.py，放到calvin-lerobot同级目录下并运行，填充count字段
训练脚本：fix_lerobot_loading.py/train_calvin.py：前者处理数据集与lerobot不适配问题，后者直接在命令行传入lerobot训练脚本相同的参数开始训练
数据集版本降级脚本：convert_dataset_v30_to_v21.py，来自：https://github.com/Tavish9/any4lerobot/tree/984be466e36a3c451814e6f8a71d1ee6ca3cdfb3/ds_version_convert/v30_to_v21
splitABC合并流程：官方0.5.x版本lerobot升级数据集到v3.0 → 官方0.5.x版本lerobot合并数据集 → 使用降级脚本降级至v2.1。！！！该过程缓存十分占磁盘空间！！！
降级后数据的meta中info.json的以下几个字段会缺少信息需要手动补充：
        "timestamp": {
            "dtype": "float32",
            "shape": [1],
            "names": ["timestamp"],
            "fps": 10
        },
        "frame_index": {
            "dtype": "int64",
            "shape": [1],
            "names": ["frame_index"],
            "fps": 10
        },
        "episode_index": {
            "dtype": "int64",
            "shape": [1],
            "names": ["episode_index"],
            "fps": 10
        },
        "index": {
            "dtype": "int64",
            "shape": [1],
            "names": ["index"],
            "fps": 10
        },
        "task_index": {
            "dtype": "int64",
            "shape": [1],
            "names": ["task_index"],
            "fps": 10
        },
