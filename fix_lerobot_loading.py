#!/usr/bin/env python3
"""
修复 LeRobot 加载 calvin-lerobot 数据集的兼容性问题。

一次性解决以下差异:
1. parquet 中图像以 HF Image extension type ({'bytes':..., 'path':...}) 存储 → 自定义解码
2. episodes_stats.jsonl 缺少 count 字段 → 需预先运行 fix_dataset_stats.py
3. CALVIN 用 "actions"(复数) / "state"，lerobot 期望 "action"(单数) / "observation.state"
4. 图像特征 (image, wrist_image) 缺少 stats → 自动补空 dict
5. Dtype: parquet 中 float 数据可能为 float64 → 转为 float32

用法:
    from fix_lerobot_loading import fix_lerobot_loading
    fix_lerobot_loading()
"""

import io
import numpy as np
import PIL.Image
import torch
from torchvision import transforms

# ============================================================================
# 模块级函数（可 pickle，兼容 DataLoader num_workers > 0）
# ============================================================================

def _alias(d, old_key, new_key):
    """如果 old_key 存在且 new_key 不存在，添加别名"""
    if old_key in d and new_key not in d:
        d[new_key] = d[old_key]


def _decode_hf_image(item: dict) -> torch.Tensor:
    """解码 HF Image extension type ({'bytes': ..., 'path': None})"""
    if item.get("bytes") is not None:
        img = PIL.Image.open(io.BytesIO(item["bytes"])).convert("RGB")
    elif item.get("path") is not None:
        img = PIL.Image.open(item["path"]).convert("RGB")
    else:
        raise ValueError(f"Cannot decode image from item: {list(item.keys())}")
    return transforms.ToTensor()(img)  # (C,H,W), float32, [0,1]


def custom_hf_transform_to_torch(items_dict: dict) -> dict:
    """
    替代 lerobot 的 hf_transform_to_torch：
    - 解码 dict 格式的 HF Image
    - 浮点数据转为 float32
    - 整数数据保持 int64
    - 列名不变（保留 "actions" 复数形式）
    """
    to_tensor = transforms.ToTensor()
    for key in items_dict:
        first_item = items_dict[key][0]
        if isinstance(first_item, PIL.Image.Image):
            items_dict[key] = [to_tensor(img) for img in items_dict[key]]
        elif isinstance(first_item, dict) and "bytes" in first_item:
            items_dict[key] = [_decode_hf_image(item) for item in items_dict[key]]
        elif first_item is None:
            pass
        else:
            converted = []
            for x in items_dict[key]:
                if isinstance(x, str):
                    converted.append(x)
                else:
                    arr = np.array(x)
                    if np.issubdtype(arr.dtype, np.floating):
                        converted.append(torch.tensor(arr, dtype=torch.float32))
                    else:
                        converted.append(torch.tensor(arr))
            items_dict[key] = converted
    return items_dict

# ============================================================================
# 补丁
# ============================================================================

def fix_lerobot_loading(verbose: bool = True):
    """应用所有兼容性补丁"""

    # --- P1: 加载 parquet 时不传 features（避免列名不匹配），用自定义 transform 解码图像 ---
    _patch_load_hf_dataset()

    # --- P2: 替换 hf_transform_to_torch ---
    import lerobot.datasets.utils as ds_utils
    ds_utils.hf_transform_to_torch = custom_hf_transform_to_torch

    # --- P3: 策略特征映射 (actions→action, state→STATE) ---
    _patch_dataset_to_policy_features()

    # --- P4: delta_timestamps 匹配 actions（复数）---
    _patch_resolve_delta_timestamps()

    # --- P5: stats 补全（image/wrist_image 空 dict + action 别名）---
    _patch_load_metadata()

    # --- P6+P7: batch 别名 + episode_index 自动修正 ---
    _patch_getitem_add_action_alias()

    # --- P8: 合并 _query_hf_dataset 的冗余 .select() 调用 ---
    _patch_query_hf_dataset()

    if verbose:
        print("[fix_lerobot_loading] All 8 patches applied successfully")


# ============================================================================
# P1: load_hf_dataset — 不用 features schema，绕过列名检查
# ============================================================================

def _patch_load_hf_dataset():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from datasets import load_dataset

    _original_load = LeRobotDataset.load_hf_dataset

    def patched_load_hf_dataset(self):
        if self.episodes is None:
            path = str(self.root / "data")
            hf_dataset = load_dataset("parquet", data_dir=path, split="train")
        else:
            files = [str(self.root / self.meta.get_data_file_path(ep_idx))
                     for ep_idx in self.episodes]
            hf_dataset = load_dataset("parquet", data_files=files, split="train")
        hf_dataset.set_transform(custom_hf_transform_to_torch)
        return hf_dataset

    LeRobotDataset.load_hf_dataset = patched_load_hf_dataset


# ============================================================================
# P3: dataset_to_policy_features — CALVIN 键名映射
# ============================================================================

def _patch_dataset_to_policy_features():
    from lerobot.datasets import utils as ds_utils
    from lerobot.configs.types import FeatureType, PolicyFeature

    def patched(features):
        policy_features = {}
        for key, ft in features.items():
            shape = ft["shape"]
            if ft["dtype"] in ["image", "video"]:
                type_ = FeatureType.VISUAL
                if len(shape) != 3:
                    raise ValueError(f"Number of dimensions of {key} != 3 (shape={shape})")
                names = ft["names"]
                if names[2] in ["channel", "channels"]:
                    shape = (shape[2], shape[0], shape[1])
            elif key == "observation.environment_state":
                type_ = FeatureType.ENV
            elif key.startswith("observation"):
                type_ = FeatureType.STATE
            elif key.startswith("action"):
                key = "action"   # actions → action (lerobot 期望单数)
                type_ = FeatureType.ACTION
            elif key == "state":
                type_ = FeatureType.STATE
            else:
                continue

            policy_features[key] = PolicyFeature(type=type_, shape=shape)

        return policy_features

    ds_utils.dataset_to_policy_features = patched


# ============================================================================
# P4: resolve_delta_timestamps — 匹配 actions（复数）
# ============================================================================

def _patch_resolve_delta_timestamps():
    from lerobot.datasets import factory as ds_factory

    def patched(cfg, ds_meta):
        from lerobot.configs.policies import PreTrainedConfig
        delta_timestamps = {}
        fps = ds_meta.fps
        for key in ds_meta.features:
            if (key == "action" or key == "actions") and cfg.action_delta_indices is not None:
                delta_timestamps[key] = [i / fps for i in cfg.action_delta_indices]
            if key == "next.reward" and cfg.reward_delta_indices is not None:
                delta_timestamps[key] = [i / fps for i in cfg.reward_delta_indices]
            if key.startswith("observation.") and cfg.observation_delta_indices is not None:
                delta_timestamps[key] = [i / fps for i in cfg.observation_delta_indices]
        return delta_timestamps if len(delta_timestamps) > 0 else None

    ds_factory.resolve_delta_timestamps = patched


# ============================================================================
# P8: _query_hf_dataset — 合并冗余 .select() 调用，提速数据加载
#     原代码对每个 key 独立调用 self.hf_dataset.select(q_idx)，
#     每步 6+ 次创建/销毁 Dataset 对象。改为按相同 q_idx 分组，
#     .select() 一次后提取所有列，减少 4-6 倍开销。
# ============================================================================

def _patch_query_hf_dataset():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    _original = LeRobotDataset._query_hf_dataset

    def patched_query_hf_dataset(self, query_indices):
        # 按相同的 query indices 分组
        idx_to_keys = {}
        for key, q_idx in query_indices.items():
            if key in self.meta.video_keys:
                continue
            idx_tuple = tuple(q_idx)
            idx_to_keys.setdefault(idx_tuple, []).append(key)

        result = {}
        for idx_tuple, keys in idx_to_keys.items():
            q_idx = list(idx_tuple)
            selected = self.hf_dataset.select(q_idx)
            for key in keys:
                result[key] = torch.stack(selected[key])
        return result

    LeRobotDataset._query_hf_dataset = patched_query_hf_dataset


# ============================================================================
# P7: _get_query_indices — 修正合并数据集的局部 episode_index
#     合并后 parquet 内部仍是原 split 的局部 episode_index，
#     但 episode_data_index 按全局编号，导致边界判断错误。
#     根据全局 frame index (idx) 反查正确的 episode_index。
# ============================================================================

def _patch_get_query_indices(dataset_cls):
    _original_get_query_indices = dataset_cls._get_query_indices

    def patched_get_query_indices(self, idx, ep_idx):
        # 用全局 frame index 计算正确的 episode_index
        # episode_data_index["to"] 是 cumulative frame counts
        to = self.episode_data_index["to"]
        corrected_ep = int(torch.searchsorted(to, idx, right=True).item())
        return _original_get_query_indices(self, idx, corrected_ep)

    dataset_cls._get_query_indices = patched_get_query_indices


# ============================================================================
# P6: __getitem__ — batch 中 actions → action 别名
#     不能放在 transform 里（会导致 format_column 单列查询失败）
#     在 __getitem__ 返回前补上 action 别名即可
# ============================================================================

def _patch_getitem_add_action_alias():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    _original_getitem = LeRobotDataset.__getitem__

    def patched_getitem(self, idx):
        item = _original_getitem(self, idx)
        # CALVIN → lerobot 键名别名（同时覆盖 delta_indices 产生的 _is_pad 键）
        _alias(item, "actions", "action")
        _alias(item, "actions_is_pad", "action_is_pad")
        _alias(item, "state", "observation.state")
        _alias(item, "image", "observation.images.image")
        _alias(item, "wrist_image", "observation.images.wrist_image")
        return item

    # ---- 同时修补 _get_query_indices 以修正 parquet 内过期的局部 episode_index ----
    _patch_get_query_indices(LeRobotDataset)

    LeRobotDataset.__getitem__ = patched_getitem


# ============================================================================
# P5: load_metadata — stats 补全
# ============================================================================

def _patch_load_metadata():
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    _original = LeRobotDatasetMetadata.load_metadata

    def patched(self):
        _original(self)
        # 为图像特征补空 stats（后续 make_dataset 会填入 ImageNet stats）
        for key in self.camera_keys:
            if key not in self.stats:
                self.stats[key] = {}
        # 为 actions 添加 action 别名（策略归一化模块查找 "action"）
        if "actions" in self.stats and "action" not in self.stats:
            self.stats["action"] = self.stats["actions"]
        # 同样处理 episodes_stats（每个 episode 的统计也需要别名）
        for ep_stats in self.episodes_stats.values():
            if "actions" in ep_stats and "action" not in ep_stats:
                ep_stats["action"] = ep_stats["actions"]

    LeRobotDatasetMetadata.load_metadata = patched
