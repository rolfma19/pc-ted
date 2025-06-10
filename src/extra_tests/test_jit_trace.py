import MinkowskiEngine as ME
import torch
import numpy as np
import torch.nn.functional as F

device = "cpu"

if __name__ =="__main__":
    coords_batch_1 = torch.tensor([[0,0],[1,1],[2,2]], dtype=torch.float32).to(device)

    feats_batch_1 = torch.tensor([[1],[1],[1]], dtype=torch.float32).to(device)
    coords, feats = ME.utils.sparse_collate([coords_batch_1], [feats_batch_1])
    test_sparse_tensor = ME.SparseTensor(features=feats, coordinates=coords, tensor_stride=2)
    kernel_size=2
    stride=2
    test_conv = ME.MinkowskiGenerativeConvolutionTranspose(1, 1, kernel_size=kernel_size, stride=stride, dimension=2).to(device)
    

    out = test_conv(test_sparse_tensor)
    # 无法使用torch.jit.trace导出ME网络，这就意味着不能直接用PNNX格式
    torch.jit.trace(test_conv, test_sparse_tensor)

