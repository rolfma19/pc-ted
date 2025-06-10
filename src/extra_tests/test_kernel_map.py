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
    # print('test_sparse_tensor:', test_sparse_tensor)
    kernel_size=2
    stride=2
    test_conv = ME.MinkowskiGenerativeConvolutionTranspose(1, 1, kernel_size=kernel_size, stride=stride, dimension=2).to(device)
    # 初始化卷积核权重为固定值
    fixed_weight_value = 1  # 固定的权重值
    with torch.no_grad():   # 在不计算梯度的情况下修改参数
        test_conv.kernel.data.fill_(fixed_weight_value)
    # print('卷积核权重:', test_conv.kernel.data)

    out = test_conv(test_sparse_tensor)
    # print('out:', out)
    iC = test_sparse_tensor.C.cpu().numpy()
    oC = out.C.cpu().numpy()
    print(iC)
    print(oC)
    print(out.F)
    kernel_maps = test_sparse_tensor.coordinate_manager.kernel_map(
            test_sparse_tensor.coordinate_map_key,out.coordinate_map_key,kernel_size=kernel_size,stride=stride
        )
    for kernel_index, in_out_map in kernel_maps.items():
            for i, o in zip(in_out_map[0], in_out_map[1]):
                print(kernel_index, iC[i], "->", oC[o])

