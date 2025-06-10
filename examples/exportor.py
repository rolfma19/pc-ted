import MinkowskiEngine as ME
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import onnx
from onnx import helper
from MinkowskiEngine.exptool import export_onnx

device = "cpu"


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = ME.MinkowskiConvolution(
            1,
            1,
            kernel_size=3,
            stride=1,
            dimension=2,
            bias=True,
        )
        self.relu1 = ME.MinkowskiReLU()
        self.conv2 = ME.MinkowskiConvolution(
            1,
            1,
            kernel_size=3,
            stride=1,
            dimension=2,
            bias=True,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        return x


if __name__ == "__main__":
    coords_batch_1 = torch.tensor([[0, 0], [1, 1], [2, 2]], dtype=torch.float32).to(
        device
    )

    feats_batch_1 = torch.tensor([[1], [1], [1]], dtype=torch.float32).to(device)
    coords, feats = ME.utils.sparse_collate([coords_batch_1], [feats_batch_1])
    x = ME.SparseTensor(features=feats, coordinates=coords, tensor_stride=2)

    model = Model()

    y = model(x)

    export_onnx(model, x, "model.onnx")
