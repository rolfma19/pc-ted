import MinkowskiEngine as ME
import torch
import numpy as np
import torch.nn.functional as F

device = "cpu"

if __name__ == "__main__":
    coords_batch_1 = torch.tensor(
        [[0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 1, 2], [1, 2, 2], [2, 2, 2]],
        dtype=torch.float32,
    ).to(device)
    feats_batch_1 = torch.tensor([[10], [20], [30], [40], [50], [60]], dtype=torch.float32).to(device)
    coords, feats = ME.utils.sparse_collate([coords_batch_1], [feats_batch_1])

    coords_batch_2 = torch.tensor(
        [[1, 1, 1], [0, 0, 2]],
        dtype=torch.float32,
    ).to(device)

    feats_batch_2 = torch.tensor([[10], [20]], dtype=torch.float32).to(device)

    query_coords, _ = ME.utils.sparse_collate([coords_batch_2], [feats_batch_2])

    input = ME.SparseTensor(
        features=feats, coordinates=coords
    )  # 准备好特征和坐标后，创建一个稀疏张量
    part_input = input.features_at_coordinates(query_coords.float())
    print(part_input)
