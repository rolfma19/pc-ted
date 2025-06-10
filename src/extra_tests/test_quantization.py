import MinkowskiEngine as ME
import torch
from torch import nn

device = "cpu"


class MEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = ME.MinkowskiConvolution(
            1,
            1,
            kernel_size=3,
            stride=1,
            dimension=3,
        )

    def forward(self, x):
        y = self.conv(x)
        return y


if __name__ == "__main__":
    coords_batch_1 = torch.tensor(
        [[0, 0, 0]],
        dtype=torch.int32,
    ).to(device)

    feats_batch_1 = torch.tensor([[1]], dtype=torch.float32).to(device)

    coords, feats = ME.utils.sparse_collate([coords_batch_1], [feats_batch_1])
    input = ME.SparseTensor(
        features=feats, coordinates=coords, tensor_stride=1
    )  # 准备好特征和坐标后，创建一个稀疏张量

    model = MEModel()
    model.eval()
    output = model(input)
