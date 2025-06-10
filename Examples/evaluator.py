import torch
import torch.nn as nn
import torch
import MinkowskiEngine as ME
from MinkowskiEngine.profiler import profile
device="cuda"

coords_batch_1 = torch.tensor([[0,0,0],[1,1,1],[2,2,2]], dtype=torch.float32).to(device)

feats_batch_1 = torch.tensor([[1,1],[1,1],[1,1]], dtype=torch.float32).to(device)
coords, feats = ME.utils.sparse_collate([coords_batch_1], [feats_batch_1])
input = ME.SparseTensor(features=feats, coordinates=coords)

class TestModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv = ME.MinkowskiConvolution(2, 16, kernel_size=3, stride=1, dimension=3).to(device)

    def forward(self, x):
        y=self.conv(x)
        return y


model=TestModel()
output=model(input)
macs, params = profile(model, inputs=(input,))
print(macs, params)