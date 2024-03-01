"""This is a small JIT compiler for micrograd computation graphs using MLIR.

The MLIR is lowered to LLVM IR and then executed using an LLVM JIT engine.
The comments in the file are meant to be liberal as this is a demonstration
and learning project.
"""

from micrograd.engine import Value
from micrograd.nn import Neuron, Layer, MLP
import mlir.dialects.stablehlo as stablehlo
import mlir.dialects.func as func
from mlir.ir import Context, Location, InsertionPoint, Module
from mlir import ir
from typing import Union, Optional
import array
import collections


# Helpers for converting to/from MLIR types
def _0d_tensor_type():
    return ir.RankedTensorType.get([],ir.F32Type.get())

def _float_to_tensor(x):
    tensor_type = _0d_tensor_type()
    val_array = bytearray(array.array('f',[float(x)]))
    return ir.DenseElementsAttr.get(val_array, type = tensor_type)


def _tensor_to_float(x):
    return float(x.get_splat_value())


class Compiler:
    """Compiler for a micrograd computation Value graph to StableHLO."""

    def __init__(self, compiled_values={}):
        self.compiled_values = compiled_values

    def walk(self, value: Value) -> ir.Value:
        """Walk the Value graph and convert it an isomorphic MLIR StableHLO dialect graph."""

        if value in self.compiled_values:
            return self.compiled_values[value]
        match value._op:
            case "":
              return stablehlo.constant(_float_to_tensor(value.data))
            case "*":
                lhs, rhs = value._prev
                return stablehlo.multiply(self.walk(lhs), self.walk(rhs))
            case "+":
                lhs, rhs = value._prev
                return stablehlo.add(self.walk(lhs), self.walk(rhs))
            case "ReLU":
                (item,) = value._prev
                return stablehlo.maximum(self.walk(Value(0.0)), self.walk(item))
        if "**" in value._op:
            base, exp = value._prev
            return stablehlo.power(self.walk(base), self.walk(exp))


def _get_args_num(net: Union[Value, Neuron, Layer, MLP]) -> int:
    if isinstance(net, Neuron):
        return len(net.parameters()) - 1
    if isinstance(net, Layer):
        return _get_args_num(net.neurons[0])
    if isinstance(net, MLP):
        return _get_args_num(net.layers[0])
    assert isinstance(net, Value)
    return 0


def _get_results_num(net: Union[Value, Neuron, Layer, MLP]) -> int:
    if isinstance(net, Layer):
        return len(net.neurons)
    if isinstance(net, MLP):
        return _get_results_num(net.layers[-1])
    assert isinstance(net, Value) or isinstance(net, Neuron)
    return 1

def _compile(net: Union[Value, Neuron, Layer, MLP]):
    """Adds the main method to a MLIR module.

    This function assumes it is called within a context and insertion point.
    """
    args_num = _get_args_num(net)
    args_types = [_0d_tensor_type()] * args_num
    args_values = [Value(0) for _ in range(args_num)]

    @func.func(*args_types)
    def main(*args):
        # This is a bit of a hack to figure out the computation graph.
        # Rather than model the various remaining types such as
        # Neuron, Layer, and MLP, we instead execute the computation
        # and since the result is a Value it encodes the whole graph.
        # This is OK since the point of JIT is to speedup subsequent
        # executions.
        net_value = net if isinstance(net, Value) else net(args_values)
        # The computation graph earlier was created with seed values of Value(0).
        # We now need to replace these with the actual arguments provided to the
        # MLIR main function.
        # We accomplish this by creating a mapping from the seed values to the
        # compiled arguments (cv). The walk method will replace the seed values
        # when traversing the graph wth the actual arguments
        compiled_values = {v: cv for v, cv in zip(args_values, args)}
        compiler = Compiler(compiled_values)
        if isinstance(net_value, list):
            return [compiler.walk(value) for value in net_value]
        return compiler.walk(net_value)

def _compile_standalone(net: Union[Value, Neuron, Layer, MLP]) -> ir.Module:
    with Context() as ctx, Location.unknown():
        stablehlo.register_dialect(ctx)
        module = Module.create()
        with InsertionPoint(module.body):
            _compile(net)
        return module

def _compile_arguments(m, args):
    with m.operation.context, Location.unknown():
        return [_float_to_tensor(v) for v in args]


class JittedNet:
    def __init__(
        self,
        net: Union[Value, Neuron, Layer, MLP],
        m: ir.Module
    ):
        self.net = net
        self.m = m

    def __call__(self, x: Optional[list[float]] = None):
        if isinstance(self.net, Value) and x != None:
            raise "You should not pass any arguments to a Value."
        xs = [] if isinstance(self.net, Value) else x

        args = _compile_arguments(self.m, xs)
        res = stablehlo.eval_module(self.m, args)
        res = [_tensor_to_float(r) for r in res]
        num_results = _get_results_num(self.net)
        assert num_results == len(res)
        return res[0] if num_results == 1 else [res[i] for i in range(num_results)]

    def __str__(self):
        return str(self.m)


def jit(net: Union[Value, Neuron, Layer, MLP]) -> JittedNet:
    """Given a micrograd computation graph, compile it to MLIR and then to LLVM.

    You can also print the returned object to see the MLIR module.

    @return: a callable that takes the input arguments of the computation graph
    """
    module = _compile_standalone(net)
    return JittedNet(net, module)
