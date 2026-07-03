import numpy as np
import torch
from torch.utils.data import Dataset

class AMBCSequenceDataset(Dataset):
    """
    标准协议 Dataset：直接读取预处理后的固定长度 numpy 数组
    """
    def __init__(self, X_path: str, y_path: str):
        self.X = np.load(X_path).astype(np.float32)   # (N, 4096, 23)
        self.y = np.load(y_path).astype(np.int64)       # (N,)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # 返回 dict，兼容 SongX-1 的模型接口
        return {
            "x": torch.from_numpy(self.X[idx]),           # (4096, 23)
            "mask": torch.ones(self.X.shape[1], dtype=torch.bool),  # (4096,) 全 True，因为无 padding
            "static": torch.zeros(0, dtype=torch.float32),  # 占位，不用静态特征
            "y": torch.tensor(self.y[idx], dtype=torch.long),
            "id": torch.tensor(idx, dtype=torch.long),       # flight-level 聚合用
        }