import MinkowskiEngine as ME
import torch
import numpy as np
import torch.nn.functional as F

device = "cpu"

if __name__ == "__main__":

    coords_batch_1 = torch.tensor(
        [
            [0, 0, 0],
        ],
        dtype=torch.float32,
    ).to(device)

    feats_batch_1 = torch.tensor(
        [
            [1],
        ],
        dtype=torch.float32,
    ).to(device)

    coords, feats = ME.utils.sparse_collate([coords_batch_1], [feats_batch_1])
    test_sparse_tensor = ME.SparseTensor(
        features=feats, coordinates=coords, tensor_stride=2
    )
    # print('test_sparse_tensor:', test_sparse_tensor)

    test_gct = ME.MinkowskiGenerativeConvolutionTranspose(1, 1, 2, 2, dimension=3).to(
        device
    )

    out = test_gct(test_sparse_tensor)
    print(out.C)
    # print('out:', out)
