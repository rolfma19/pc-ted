# Copyright (c) 2020 NVIDIA CORPORATION.
# Copyright (c) 2018-2020 Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.
import os
import torch
import warnings

from MinkowskiCommon import convert_to_int_list, StrideType
from MinkowskiEngineBackend._C import (
    CoordinateMapKey,
    CoordinateMapType,
    GPUMemoryAllocatorType,
    MinkowskiAlgorithm,
)
from MinkowskiTensor import (
    SparseTensorQuantizationMode,
    SparseTensorOperationMode,
    Tensor,
    sparse_tensor_operation_mode,
    global_coordinate_manager,
    set_global_coordinate_manager,
)
from MinkowskiCoordinateManager import CoordinateManager
from sparse_matrix_functions import (
    MinkowskiSPMMFunction,
    MinkowskiSPMMAverageFunction,
)


class SparseTensor(Tensor):
    r"""
    SparseTensor 使用 COO 格式。

    C 是其坐标，F 是其属性。
    """

    def __init__(
        self,
        features: torch.Tensor,
        coordinates: torch.Tensor = None,
        # 可选的坐标相关参数
        tensor_stride: StrideType = 1,
        coordinate_map_key: CoordinateMapKey = None,
        coordinate_manager: CoordinateManager = None,
        quantization_mode: SparseTensorQuantizationMode = SparseTensorQuantizationMode.RANDOM_SUBSAMPLE,
        # 可选的管理器相关参数
        allocator_type: GPUMemoryAllocatorType = None,
        minkowski_algorithm: MinkowskiAlgorithm = None,
        requires_grad=None,
        device=None,
    ):
        """
        :param features: 形状为 [N,C] N 是坐标点数，C 是特征维度
        :param coordinates:  形状为 [N,D+1] N 是坐标点数，D 是坐标维度，多一维是 batch 维
        :param tensor_stride: 描述 SparseTensor 中坐标的间隔，如果原来为 1，经过一次下采样后会变成 2
        :param coordinate_map_key: 描述 SparseTensor 的 D 维坐标与存储索引的映射
        :param coordinate_manager: 用来管理坐标，如 CoordinateMapKey KernelMap 等
        :param quantization_mode: 连续坐标转离散坐标时的量化方法
        :param allocator_type: 内存分配器
        :param minkowski_algorithm: 计算方法，有快速模式之类的
        :param requires_grad: 是否计算梯度
        :param device: SparseTensor 的设备
        """
        # 检查参数类型
        assert isinstance(features, torch.Tensor), "Features must be a torch.Tensor"
        assert (
            features.ndim == 2
        ), f"The feature should be a matrix, The input feature is an order-{features.ndim} tensor."
        assert isinstance(quantization_mode, SparseTensorQuantizationMode)
        self.quantization_mode = quantization_mode

        if coordinates is not None:
            assert isinstance(coordinates, torch.Tensor)
        if coordinate_map_key is not None:
            assert isinstance(coordinate_map_key, CoordinateMapKey)
            assert (
                coordinate_manager is not None
            ), "Must provide coordinate_manager if coordinate_map_key is provided"
            assert (
                coordinates is None
            ), "Must not provide coordinates if coordinate_map_key is provided"
        if coordinate_manager is not None:
            assert isinstance(coordinate_manager, CoordinateManager)
        if coordinates is None and (
            coordinate_map_key is None or coordinate_manager is None
        ):
            raise ValueError(
                "Either coordinates or (coordinate_map_key, coordinate_manager) pair must be provided."
            )

        Tensor.__init__(self)

        # 把 features 和 coordinates 放到设定的 device 上面
        if device is not None:
            features = features.to(device)
            if coordinates is not None:
                # assertion check for the map key done later
                coordinates = coordinates.to(device)

        self._D = (
            coordinates.size(1) - 1 if coordinates is not None else coordinate_manager.D
        )
        ##########################
        # 设置坐标管理器
        ##########################
        if coordinate_manager is None:
            # 如果设置成了共享坐标管理器，就使用全局坐标管理器
            if (
                sparse_tensor_operation_mode()
                == SparseTensorOperationMode.SHARE_COORDINATE_MANAGER
            ):
                coordinate_manager = global_coordinate_manager()
                if coordinate_manager is None:
                    coordinate_manager = CoordinateManager(
                        D=self._D,
                        coordinate_map_type=(
                            CoordinateMapType.CUDA
                            if coordinates.is_cuda
                            else CoordinateMapType.CPU
                        ),
                        allocator_type=allocator_type,
                        minkowski_algorithm=minkowski_algorithm,
                    )
                    set_global_coordinate_manager(coordinate_manager)
            else:
                coordinate_manager = CoordinateManager(
                    D=coordinates.size(1) - 1,
                    coordinate_map_type=(
                        CoordinateMapType.CUDA
                        if coordinates.is_cuda
                        else CoordinateMapType.CPU
                    ),
                    allocator_type=allocator_type,
                    minkowski_algorithm=minkowski_algorithm,
                )
        self._manager = coordinate_manager

        ##########################
        # Initialize coords
        ##########################
        if coordinates is not None:
            assert (
                features.shape[0] == coordinates.shape[0]
            ), "The number of rows in features and coordinates must match."

            assert (
                features.is_cuda == coordinates.is_cuda
            ), "Features and coordinates must have the same backend."
            # coordinate_map_key is a pair of stensor_stride and string_id
            coordinate_map_key = CoordinateMapKey(
                convert_to_int_list(tensor_stride, self._D), ""
            )
            # 调用 self.initialize_coordinates 初始化坐标
            coordinates, features, coordinate_map_key = self.initialize_coordinates(
                coordinates, features, coordinate_map_key
            )
        else:  # coordinate_map_key is not None:
            assert coordinate_map_key.is_key_set(), "The coordinate key must be valid."

        if requires_grad is not None:
            features.requires_grad_(requires_grad)

        self._F = features
        self._C = coordinates
        self.coordinate_map_key = coordinate_map_key
        self._batch_rows = None

    @property
    def coordinate_key(self):
        return self.coordinate_map_key

    def initialize_coordinates(self, coordinates, features, coordinate_map_key):
        """
        Initialize coordinates by putting coordinates to coordinate_manager and removing duplicate coordinates
        :param coordinates: [N, D]
        :param features: [N, C]
        :param coordinate_map_key: a pair of stensor_stride and string_id
        :return: unique_coordinates, features, coordinate_map_key
        """
        if not isinstance(coordinates, (torch.IntTensor, torch.cuda.IntTensor)):
            warnings.warn(
                "coordinates implicitly converted to torch.IntTensor. "
                + "To remove this warning, use `.int()` to convert the "
                + "coords into an torch.IntTensor"
            )
            coordinates = torch.floor(coordinates).int()  # 采用 floor 量化坐标
        # *coordinate_map_key.get_key() 实际为tensor_stride, coordinate_map_key是用 tensor_stride和 string_id 组成的字符串来标识的
        (
            coordinate_map_key,
            (unique_index, inverse_mapping),
        ) = self._manager.insert_and_map(coordinates, *coordinate_map_key.get_key())

        self.unique_index = unique_index.long()
        coordinates = coordinates[self.unique_index]

        if len(inverse_mapping) == 0:  # 没有在坐标管理器加入新坐标
            # When the input has the same shape as the output
            self.inverse_mapping = torch.arange(
                len(features),
                dtype=inverse_mapping.dtype,
                device=inverse_mapping.device,
            )
            return coordinates, features, coordinate_map_key

        self.inverse_mapping = inverse_mapping
        if self.quantization_mode == SparseTensorQuantizationMode.UNWEIGHTED_SUM:
            spmm = MinkowskiSPMMFunction()
            N = len(features)
            cols = torch.arange(
                N,
                dtype=self.inverse_mapping.dtype,
                device=self.inverse_mapping.device,
            )
            vals = torch.ones(N, dtype=features.dtype, device=features.device)
            size = torch.Size([len(self.unique_index), len(self.inverse_mapping)])
            features = spmm.apply(self.inverse_mapping, cols, vals, size, features)
        elif self.quantization_mode == SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE:
            spmm_avg = MinkowskiSPMMAverageFunction()
            N = len(features)
            cols = torch.arange(
                N,
                dtype=self.inverse_mapping.dtype,
                device=self.inverse_mapping.device,
            )
            size = torch.Size([len(self.unique_index), len(self.inverse_mapping)])
            features = spmm_avg.apply(self.inverse_mapping, cols, size, features)
        elif self.quantization_mode == SparseTensorQuantizationMode.RANDOM_SUBSAMPLE:
            features = features[self.unique_index]
        else:
            # No quantization
            pass

        return coordinates, features, coordinate_map_key

    # Conversion functions
    def sparse(self, min_coords=None, max_coords=None, contract_coords=True):
        r"""Convert the :attr:`MinkowskiEngine.SparseTensor` to a torch sparse
        tensor.

        Args:
            :attr:`min_coords` (torch.IntTensor, optional): The min
            coordinates of the output sparse tensor. Must be divisible by the
            current :attr:`tensor_stride`.

            :attr:`max_coords` (torch.IntTensor, optional): The max coordinates
            of the output sparse tensor (inclusive). Must be divisible by the
            current :attr:`tensor_stride`.

            :attr:`contract_coords` (bool, optional): Given True, the output
            coordinates will be divided by the tensor stride to make features
            contiguous.

        Returns:
            :attr:`spare_tensor` (torch.sparse.Tensor): the torch sparse tensor
            representation of the self in `[Batch Dim, Spatial Dims..., Feature
            Dim]`. The coordinate of each feature can be accessed via
            `min_coord + tensor_stride * [the coordinate of the dense tensor]`.

            :attr:`min_coords` (torch.IntTensor): the D-dimensional vector
            defining the minimum coordinate of the output sparse tensor. If
            :attr:`contract_coords` is True, the :attr:`min_coords` will also
            be contracted.

            :attr:`tensor_stride` (torch.IntTensor): the D-dimensional vector
            defining the stride between tensor elements.

        """

        if min_coords is not None:
            assert isinstance(min_coords, torch.IntTensor)
            assert min_coords.numel() == self._D
        if max_coords is not None:
            assert isinstance(max_coords, torch.IntTensor)
            assert min_coords.numel() == self._D

        def torch_sparse_Tensor(coords, feats, size=None):
            if size is None:
                if feats.dtype == torch.float64:
                    return torch.sparse.DoubleTensor(coords, feats)
                elif feats.dtype == torch.float32:
                    return torch.sparse.FloatTensor(coords, feats)
                else:
                    raise ValueError("Feature type not supported.")
            else:
                if feats.dtype == torch.float64:
                    return torch.sparse.DoubleTensor(coords, feats, size)
                elif feats.dtype == torch.float32:
                    return torch.sparse.FloatTensor(coords, feats, size)
                else:
                    raise ValueError("Feature type not supported.")

        # Use int tensor for all operations
        tensor_stride = torch.IntTensor(self.tensor_stride)

        # New coordinates
        coords = self.C
        coords, batch_indices = coords[:, 1:], coords[:, 0]

        if min_coords is None:
            min_coords, _ = coords.min(0, keepdim=True)
        elif min_coords.ndim == 1:
            min_coords = min_coords.unsqueeze(0)

        assert (
            min_coords % tensor_stride
        ).sum() == 0, "The minimum coordinates must be divisible by the tensor stride."

        if max_coords is not None:
            if max_coords.ndim == 1:
                max_coords = max_coords.unsqueeze(0)
            assert (
                max_coords % tensor_stride
            ).sum() == 0, (
                "The maximum coordinates must be divisible by the tensor stride."
            )

        coords -= min_coords

        if coords.ndim == 1:
            coords = coords.unsqueeze(1)
        if batch_indices.ndim == 1:
            batch_indices = batch_indices.unsqueeze(1)

        # return the contracted tensor
        if contract_coords:
            coords = coords // tensor_stride
            if max_coords is not None:
                max_coords = max_coords // tensor_stride
            min_coords = min_coords // tensor_stride

        new_coords = torch.cat((batch_indices, coords), dim=1).long()

        size = None
        if max_coords is not None:
            size = max_coords - min_coords + 1  # inclusive
            # Squeeze to make the size one-dimensional
            size = size.squeeze()

            max_batch = max(self._manager.get_batch_indices())
            size = torch.Size([max_batch + 1, *size, self.F.size(1)])

        sparse_tensor = torch_sparse_Tensor(
            new_coords.t().to(self.F.device), self.F, size
        )
        tensor_stride = torch.IntTensor(self.tensor_stride)
        return sparse_tensor, min_coords, tensor_stride

    def dense(self, shape=None, min_coordinate=None, contract_stride=True):
        r"""Convert the :attr:`MinkowskiEngine.SparseTensor` to a torch dense
        tensor.

        Args:
            :attr:`shape` (torch.Size, optional): The size of the output tensor.

            :attr:`min_coordinate` (torch.IntTensor, optional): The min
            coordinates of the output sparse tensor. Must be divisible by the
            current :attr:`tensor_stride`. If 0 is given, it will use the origin for the min coordinate.

            :attr:`contract_stride` (bool, optional): The output coordinates
            will be divided by the tensor stride to make features spatially
            contiguous. True by default.

        Returns:
            :attr:`tensor` (torch.Tensor): the torch tensor with size `[Batch
            Dim, Feature Dim, Spatial Dim..., Spatial Dim]`. The coordinate of
            each feature can be accessed via `min_coordinate + tensor_stride *
            [the coordinate of the dense tensor]`.

            :attr:`min_coordinate` (torch.IntTensor): the D-dimensional vector
            defining the minimum coordinate of the output tensor.

            :attr:`tensor_stride` (torch.IntTensor): the D-dimensional vector
            defining the stride between tensor elements.

        """
        if min_coordinate is not None:
            assert isinstance(min_coordinate, torch.IntTensor)
            assert min_coordinate.numel() == self._D
        if shape is not None:
            assert isinstance(shape, torch.Size)
            assert len(shape) == self._D + 2  # batch and channel
            if shape[1] != self._F.size(1):
                shape = torch.Size([shape[0], self._F.size(1), *[s for s in shape[2:]]])

        # Exception handling for empty tensor
        if self.__len__() == 0:
            assert shape is not None, "shape is required to densify an empty tensor"
            return (
                torch.zeros(shape, dtype=self.dtype, device=self.device),
                torch.zeros(self._D, dtype=torch.int32, device=self.device),
                self.tensor_stride,
            )

        # Use int tensor for all operations
        tensor_stride = torch.IntTensor(self.tensor_stride).to(self.device)

        # New coordinates
        batch_indices = self.C[:, 0]

        if min_coordinate is None:
            min_coordinate, _ = self.C.min(0, keepdim=True)
            min_coordinate = min_coordinate[:, 1:]
            if not torch.all(min_coordinate >= 0):
                raise ValueError(
                    f"Coordinate has a negative value: {min_coordinate}. Please provide min_coordinate argument"
                )
            coords = self.C[:, 1:]
        elif isinstance(min_coordinate, int) and min_coordinate == 0:
            coords = self.C[:, 1:]
        else:
            min_coordinate = min_coordinate.to(self.device)
            if min_coordinate.ndim == 1:
                min_coordinate = min_coordinate.unsqueeze(0)
            coords = self.C[:, 1:] - min_coordinate

        assert (
            min_coordinate % tensor_stride
        ).sum() == 0, "The minimum coordinates must be divisible by the tensor stride."

        if coords.ndim == 1:
            coords = coords.unsqueeze(1)

        # return the contracted tensor
        if contract_stride:
            coords = coords // tensor_stride

        nchannels = self.F.size(1)
        if shape is None:
            size = coords.max(0)[0] + 1
            shape = torch.Size(
                [batch_indices.max() + 1, nchannels, *size.cpu().numpy()]
            )

        dense_F = torch.zeros(shape, dtype=self.dtype, device=self.device)

        tcoords = coords.t().long()
        batch_indices = batch_indices.long()
        exec(
            "dense_F[batch_indices, :, "
            + ", ".join([f"tcoords[{i}]" for i in range(len(tcoords))])
            + "] = self.F"
        )

        tensor_stride = torch.IntTensor(self.tensor_stride)
        return dense_F, min_coordinate, tensor_stride

    def interpolate(self, X):
        from MinkowskiTensorField import TensorField

        assert isinstance(X, TensorField)
        if self.coordinate_map_key in X._splat:
            tensor_map, field_map, weights, size = X._splat[self.coordinate_map_key]
            size = torch.Size([size[1], size[0]])  # transpose
            features = MinkowskiSPMMFunction().apply(
                field_map, tensor_map, weights, size, self._F
            )
        else:
            features = self.features_at_coordinates(X.C)
        return TensorField(
            features=features,
            coordinate_field_map_key=X.coordinate_field_map_key,
            coordinate_manager=X.coordinate_manager,
        )

    def slice(self, X):
        r"""

        Args:
           :attr:`X` (:attr:`MinkowskiEngine.SparseTensor`): a sparse tensor
           that discretized the original input.

        Returns:
           :attr:`tensor_field` (:attr:`MinkowskiEngine.TensorField`): the
           resulting tensor field contains features on the continuous
           coordinates that generated the input X.

        Example::

           >>> # coords, feats from a data loader
           >>> print(len(coords))  # 227742
           >>> tfield = ME.TensorField(coordinates=coords, features=feats, quantization_mode=SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)
           >>> print(len(tfield))  # 227742
           >>> sinput = tfield.sparse() # 161890 quantization results in fewer voxels
           >>> soutput = MinkUNet(sinput)
           >>> print(len(soutput))  # 161890 Output with the same resolution
           >>> ofield = soutput.slice(tfield)
           >>> assert isinstance(ofield, ME.TensorField)
           >>> len(ofield) == len(coords)  # recovers the original ordering and length
           >>> assert isinstance(ofield.F, torch.Tensor)  # .F returns the features
        """
        # Currently only supports unweighted slice.
        assert X.quantization_mode in [
            SparseTensorQuantizationMode.RANDOM_SUBSAMPLE,
            SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
        ], "slice only available for sparse tensors with quantization RANDOM_SUBSAMPLE or UNWEIGHTED_AVERAGE"

        from MinkowskiTensorField import TensorField

        if isinstance(X, TensorField):
            return TensorField(
                self.F[X.inverse_mapping(self.coordinate_map_key).long()],
                coordinate_field_map_key=X.coordinate_field_map_key,
                coordinate_manager=X.coordinate_manager,
                quantization_mode=X.quantization_mode,
            )
        elif isinstance(X, SparseTensor):
            inv_map = X.inverse_mapping
            assert (
                X.coordinate_map_key == self.coordinate_map_key
            ), "Slice can only be applied on the same coordinates (coordinate_map_key)"
            return TensorField(
                self.F[inv_map],
                coordinates=self.C[inv_map],
                coordinate_manager=self.coordinate_manager,
                quantization_mode=self.quantization_mode,
            )
        else:
            raise ValueError(
                "Invalid input. The input must be an instance of TensorField or SparseTensor."
            )

    def cat_slice(self, X):
        r"""

        Args:
           :attr:`X` (:attr:`MinkowskiEngine.SparseTensor`): a sparse tensor
           that discretized the original input.

        Returns:
           :attr:`tensor_field` (:attr:`MinkowskiEngine.TensorField`): the
           resulting tensor field contains the concatenation of features on the
           original continuous coordinates that generated the input X and the
           self.

        Example::

           >>> # coords, feats from a data loader
           >>> print(len(coords))  # 227742
           >>> sinput = ME.SparseTensor(coordinates=coords, features=feats, quantization_mode=SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)
           >>> print(len(sinput))  # 161890 quantization results in fewer voxels
           >>> soutput = network(sinput)
           >>> print(len(soutput))  # 161890 Output with the same resolution
           >>> ofield = soutput.cat_slice(sinput)
           >>> assert soutput.F.size(1) + sinput.F.size(1) == ofield.F.size(1)  # concatenation of features
        """
        # Currently only supports unweighted slice.
        assert X.quantization_mode in [
            SparseTensorQuantizationMode.RANDOM_SUBSAMPLE,
            SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
        ], "slice only available for sparse tensors with quantization RANDOM_SUBSAMPLE or UNWEIGHTED_AVERAGE"

        from MinkowskiTensorField import TensorField

        inv_map = X.inverse_mapping(self.coordinate_map_key)
        features = torch.cat((self.F[inv_map], X.F), dim=1)
        if isinstance(X, TensorField):
            return TensorField(
                features,
                coordinate_field_map_key=X.coordinate_field_map_key,
                coordinate_manager=X.coordinate_manager,
                quantization_mode=X.quantization_mode,
            )
        elif isinstance(X, SparseTensor):
            assert (
                X.coordinate_map_key == self.coordinate_map_key
            ), "Slice can only be applied on the same coordinates (coordinate_map_key)"
            return TensorField(
                features,
                coordinates=self.C[inv_map],
                coordinate_manager=self.coordinate_manager,
                quantization_mode=self.quantization_mode,
            )
        else:
            raise ValueError(
                "Invalid input. The input must be an instance of TensorField or SparseTensor."
            )

    def features_at_coordinates(self, query_coordinates: torch.Tensor):
        r"""Extract features at the specified continuous coordinate matrix.

        Args:
           :attr:`query_coordinates` (:attr:`torch.FloatTensor`): a coordinate
           matrix of size :math:`N \times (D + 1)` where :math:`D` is the size
           of the spatial dimension.

        Returns:
           :attr:`queried_features` (:attr:`torch.Tensor`): a feature matrix of
           size :math:`N \times D_F` where :math:`D_F` is the number of
           channels in the feature. For coordinates not present in the current
           sparse tensor, corresponding feature rows will be zeros.
        """
        from MinkowskiInterpolation import MinkowskiInterpolationFunction

        assert (
            self.dtype == query_coordinates.dtype
        ), "Invalid query_coordinates dtype. use {self.dtype}"

        assert (
            query_coordinates.device == self.device
        ), "query coordinates device ({query_coordinates.device}) does not match the sparse tensor device ({self.device})."
        return MinkowskiInterpolationFunction().apply(
            self._F,
            query_coordinates,
            self.coordinate_map_key,
            self.coordinate_manager,
        )[0]

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + os.linesep
            + "  coordinates="
            + str(self.C)
            + os.linesep
            + "  features="
            + str(self.F)
            + os.linesep
            + "  coordinate_map_key="
            + str(self.coordinate_map_key)
            + os.linesep
            + "  coordinate_manager="
            + str(self._manager)
            + "  spatial dimension="
            + str(self._D)
            + ")"
        )

    __slots__ = (
        "_C",
        "_F",
        "_D",
        "coordinate_map_key",
        "_manager",
        "unique_index",
        "inverse_mapping",
        "quantization_mode",
        "_batch_rows",
    )


def _get_coordinate_map_key(
    input: SparseTensor,
    coordinates: torch.Tensor = None,
    tensor_stride: StrideType = 1,
    expand_coordinates: bool = False,
):
    r"""Returns the coordinates map key."""
    if coordinates is not None and not expand_coordinates: # there are target coordinates
        assert isinstance(coordinates, (CoordinateMapKey, torch.Tensor, SparseTensor))
        if isinstance(coordinates, torch.Tensor):
            assert coordinates.ndim == 2
            coordinate_map_key = CoordinateMapKey(
                convert_to_int_list(tensor_stride, coordinates.size(1) - 1), ""
            )

            (
                coordinate_map_key,
                (unique_index, inverse_mapping),
            ) = input._manager.insert_and_map(
                coordinates, *coordinate_map_key.get_key()
            )
        elif isinstance(coordinates, SparseTensor):
            coordinate_map_key = coordinates.coordinate_map_key
        else:  # CoordinateMapKey type due to the previous assertion
            coordinate_map_key = coordinates
    else:  # coordinates is None
        coordinate_map_key = CoordinateMapKey(
            input.coordinate_map_key.get_coordinate_size()
        ) # only set the m_coordinates_size in CoordinateMapKey
    return coordinate_map_key
