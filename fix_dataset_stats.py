#!/usr/bin/env python3
"""
修复 calvin-lerobot 数据集的 episodes_stats.jsonl
问题: 每个特征的统计信息缺少 'count' 字段，导致 lerobot 的 aggregate_stats 崩溃。

对每个 split (splitA, splitB, splitC, splitD)：
1. 读取 episodes.jsonl 获取每个 episode 的 length
2. 读取 episodes_stats.jsonl 获取现有统计
3. 为每个特征添加缺失的 count 字段
4. 写回修复后的 episodes_stats.jsonl
"""

import json
import sys
import copy
from pathlib import Path

# 从 lerobot 导入 estimate_num_samples
try:
    from lerobot.datasets.compute_stats import estimate_num_samples
except ImportError:
    # 如果无法导入，手动实现
    def estimate_num_samples(
        dataset_len: int, min_num_samples: int = 100, max_num_samples: int = 10_000, power: float = 0.75
    ) -> int:
        if dataset_len < min_num_samples:
            min_num_samples = dataset_len
        return max(min_num_samples, min(int(dataset_len**power), max_num_samples))


# 图像类型的特征键（需要采样计算统计量）
IMAGE_KEYS = {"image", "wrist_image"}


def fix_split_stats(split_dir: Path) -> bool:
    """修复单个 split 的 episodes_stats.jsonl"""
    meta_dir = split_dir / "meta"
    episodes_path = meta_dir / "episodes.jsonl"
    stats_path = meta_dir / "episodes_stats.jsonl"

    if not episodes_path.exists():
        print(f"  [跳过] episodes.jsonl 不存在: {episodes_path}")
        return False
    if not stats_path.exists():
        print(f"  [跳过] episodes_stats.jsonl 不存在: {stats_path}")
        return False

    # 1. 读取所有 episode 的 length
    episode_lengths = {}
    with open(episodes_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ep = json.loads(line)
            episode_lengths[ep["episode_index"]] = ep["length"]

    if not episode_lengths:
        print(f"  [错误] episodes.jsonl 为空")
        return False

    # 2. 读取所有 stats
    stats_entries = []
    with open(stats_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            stats_entries.append(json.loads(line))

    # 3. 检查是否需要修复
    needs_fix = False
    for entry in stats_entries:
        for feature_key, feature_stats in entry["stats"].items():
            if "count" not in feature_stats:
                needs_fix = True
                break
        if needs_fix:
            break

    if not needs_fix:
        print(f"  [已就绪] 无需修复")
        return True

    # 4. 添加 count 字段并写回
    fixed_count = 0
    for entry in stats_entries:
        ep_idx = entry["episode_index"]
        ep_length = episode_lengths.get(ep_idx, 1)

        for feature_key, feature_stats in entry["stats"].items():
            if "count" in feature_stats:
                continue  # 已有 count，跳过
            if feature_key in IMAGE_KEYS:
                # 图像特征使用采样数量（必须是 list 格式，转 numpy 后为 shape (1,)）
                feature_stats["count"] = [estimate_num_samples(ep_length)]
            else:
                # 非图像特征使用全部帧数（必须是 list 格式，转 numpy 后为 shape (1,)）
                feature_stats["count"] = [ep_length]
            fixed_count += 1

    # 5. 写回（覆盖前备份）
    backup_path = stats_path.with_suffix(".jsonl.bak")
    if not backup_path.exists():
        import shutil
        shutil.copy2(stats_path, backup_path)
        print(f"  [备份] {backup_path}")

    with open(stats_path, "w", encoding="utf-8") as f:
        for entry in stats_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"  [修复] 添加了 {fixed_count} 个 count 字段 ({len(stats_entries)} 个 episodes)")
    return True


def main():
    base_dir = Path(__file__).parent / "calvin-lerobot"
    splits = ["splitA", "splitB", "splitC", "splitD"]

    print("=" * 60)
    print("开始修复数据集 stats...")
    print("=" * 60)

    all_ok = True
    for split_name in splits:
        split_dir = base_dir / split_name
        print(f"\n处理 {split_name}:")
        try:
            ok = fix_split_stats(split_dir)
            if not ok:
                all_ok = False
        except Exception as e:
            print(f"  [错误] {e}")
            all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print("所有 split 修复完成!")
    else:
        print("部分 split 修复失败，请看上面错误信息。")
    print("=" * 60)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
