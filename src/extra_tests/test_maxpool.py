import MinkowskiEngine as ME
import torch
import numpy as np
import torch.nn.functional as F

device = "cpu"

if __name__ == "__main__":

    coords_batch_1 = torch.tensor(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
            [0, 0, 0],
        ],
        dtype=torch.float32,
    ).to(device)

    feats_batch_1 = torch.tensor(
        [[0], [0], [0], [0], [0], [0], [0], [0], [0]], dtype=torch.float32
    ).to(device)

    coords, feats = ME.utils.sparse_collate([coords_batch_1], [feats_batch_1])
    test_sparse_tensor = ME.SparseTensor(features=feats, coordinates=coords)
    # print('test_sparse_tensor:', test_sparse_tensor)

    test_max_pool = ME.MinkowskiMaxPooling(2, 2, dimension=3).to(device)

    out = test_max_pool(test_sparse_tensor)
    print(out)
    # print('out:', out)
