import torch
import onnx
import onnx.helper as helper
import numpy as np
import MinkowskiEngine as ME


avoid_reuse_container = []
obj_to_tensor_id = {}
nodes = []
initializers = []
enable_trace = False


def register_node(fn):

    fnnames = fn.split(".")
    fn_module = eval(".".join(fnnames[:-1]))
    fn_name = fnnames[-1]
    oldfn = getattr(fn_module, fn_name)

    def make_hook(bind_fn):

        ilayer = 0

        def internal_forward(self, *args):
            global enable_trace

            if not enable_trace:
                return oldfn(self, *args)

            global avoid_reuse_container
            nonlocal ilayer

            # Use the enable_trace flag to avoid internal trace calls
            enable_trace = False
            y = oldfn(self, *args)
            bind_fn(self, ilayer, y, *args)
            enable_trace = True

            avoid_reuse_container.extend(list(args) + [y])
            ilayer += 1
            return y

        setattr(fn_module, fn_name, internal_forward)

    return make_hook


@register_node("ME.MinkowskiConvolution.forward")
def symbolic_MinkowskiConvolution(self, ilayer, y, x):
    register_tensor(y)
    print(
        f"   --> MinkowskiConvolution{ilayer} -> Input {get_tensor_id(x)}, Output {get_tensor_id(y)}"
    )

    inputs = [
        get_tensor_id(x),
        append_initializer(self.kernel.data, f"MinkowskiConvolution{ilayer}.weight"),
    ]
    if self.bias is not None:
        inputs.append(
            append_initializer(self.bias.data, f"MinkowskiConvolution{ilayer}.bias")
        )
    nodes.append(
        helper.make_node(
            "MinkowskiConvolution",
            inputs,
            [get_tensor_id(y)],
            f"MinkowskiConvolution{ilayer}",
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            dilation=self.dilation,
            has_bias=self.has_bias,
            is_transpose=self.is_transpose,
            expand_coordinates=self.expand_coordinates,
            dimension=self.dimension,
        )
    )


@register_node("ME.MinkowskiReLU.forward")
def symbolic_MinkowskiReLU(self, ilayer, y, x):
    register_tensor(y)
    print(
        f"   --> MinkowskiReLU{ilayer} -> Input {get_tensor_id(x)}, Output {get_tensor_id(y)}"
    )
    nodes.append(
        helper.make_node(
            "MinkowskiReLU",
            [get_tensor_id(x)],
            [get_tensor_id(y)],
            f"MinkowskiReLU{ilayer}",
        )
    )


def append_initializer(value, name):
    initializers.append(
        helper.make_tensor(
            name=name,
            data_type=helper.TensorProto.DataType.FLOAT,
            dims=list(value.shape),
            vals=value.cpu().data.numpy().astype(np.float32).tobytes(),
            raw=True,
        )
    )
    return name


def __obj_to_id(obj):
    idd = id(obj)
    if isinstance(obj, ME.SparseTensor):
        idd = id(obj.features)
    return idd


def set_obj_idd_assame(a_already_has_idd, b_no_idd):
    global obj_to_tensor_id
    aidd = __obj_to_id(a_already_has_idd)
    bidd = __obj_to_id(b_no_idd)

    assert aidd in obj_to_tensor_id, "A is not in tensor map"
    assert bidd not in obj_to_tensor_id, "B is already in tensor map"
    obj_to_tensor_id[bidd] = obj_to_tensor_id[aidd]


def register_tensor(obj):
    global obj_to_tensor_id
    obj_to_tensor_id[__obj_to_id(obj)] = str(len(obj_to_tensor_id))


def get_tensor_id(obj):
    idd = __obj_to_id(obj)
    assert (
        idd in obj_to_tensor_id
    ), "ops!!!😮 Cannot find the tensorid of this object. this means that some operators are not being traced. You need to confirm it."
    return obj_to_tensor_id[idd]


def export_onnx(model, x, save_onnx):

    global avoid_reuse_container, tensor_map, nodes, initializers, enable_trace
    avoid_reuse_container = []
    tensor_map = {}
    nodes = []
    initializers = []

    print("Tracing model inference...")
    print("> Do inference...")
    with torch.no_grad():
        register_tensor(x)
        enable_trace = True
        y = model(x)
        enable_trace = False

    print("Tracing done!")

    inputs = [
        helper.make_value_info(
            name="0",
            type_proto=helper.make_tensor_type_proto(
                elem_type=helper.TensorProto.DataType.FLOAT, shape=x.size()
            ),
        )
    ]

    outputs = [
        helper.make_value_info(
            name=get_tensor_id(y),
            type_proto=helper.make_tensor_type_proto(
                elem_type=helper.TensorProto.DataType.FLOAT, shape=y.size()
            ),
        )
    ]

    graph = helper.make_graph(
        name="MinkowskiNet",
        inputs=inputs,
        outputs=outputs,
        nodes=nodes,
        initializer=initializers,
    )

    model = helper.make_model(graph, producer_name="MinkowskiEngineNext")
    onnx.save_model(model, save_onnx)
    print(f"The export is completed. ONNX save as {save_onnx}, Have a nice day~")

    # clean memory
    avoid_reuse_container = []
    tensor_map = {}
    nodes = []
    initializers = []
