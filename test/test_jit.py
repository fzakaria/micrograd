import math
import random
import struct
import subprocess
import timeit
import tempfile
from micrograd.engine import Value
from micrograd.nn import Neuron, Layer, MLP
from micrograd.jit import jit

# helps investigate segmentation faults
import faulthandler

faulthandler.enable()


def test_value():
    a = Value(4.0)
    b = Value(2.0)
    c = a + b  # 6.
    d = a + c  # 10.
    jd = jit(d)
    assert math.isclose(d.data, jd(), abs_tol=1e-04)


def test_neuron():
    n = Neuron(nin=1, nonlin=False)
    n.w = [2.0]
    jn = jit(n)
    args = [10.0]
    assert math.isclose(n(args).data, jn(args), abs_tol=1e-04)


def test_layer():
    random.seed(10)
    l = Layer(nin=2, nout=1)
    jl = jit(l)
    args = [-30.0, -20.0]
    assert math.isclose(l(args).data, jl(args), abs_tol=1e-04)


def test_layer_multiple_out():
    random.seed(10)
    l = Layer(nin=2, nout=2)
    jl = jit(l)
    args = [-30.0, -20.0]
    for r, jr in zip(l(args), jl(args)):
        assert math.isclose(r.data, jr, abs_tol=1e-04)


def test_mlp():
    random.seed(10)
    nn = MLP(nin=2, nouts=[1])
    jnn = jit(nn)
    args = [-30.0, -20.0]
    assert math.isclose(nn(args).data, jnn(args), abs_tol=1e-04)


def test_mlp_complex():
    random.seed(10)
    nn = MLP(nin=2, nouts=[2, 1])
    jnn = jit(nn)
    args = [-30.0, -20.0]
    assert math.isclose(nn(args).data, jnn(args), abs_tol=1e-04)


def test_mlp_complex_multiple_out():
    random.seed(10)
    nn = MLP(nin=2, nouts=[2, 2])
    jnn = jit(nn)
    args = [-30.0, -20.0]
    for r, jr in zip(nn(args), jnn(args)):
        assert math.isclose(r.data, jr, abs_tol=1e-04)


def test_mlp_performance():
    random.seed(10)
    nn = MLP(nin=10, nouts=[30, 20, 10, 1])
    args = random.sample(range(-100, 100), 10)
    jnn = jit(nn)

    def slow_inference():
        return nn(args)

    def fast_inference():
        return jnn(args)

    slow_inference_time = timeit.timeit(slow_inference, number=1000)
    fast_inference_time = timeit.timeit(fast_inference, number=1000)
    print(f"\nslow: {slow_inference_time}\nfast: {fast_inference_time}")
    assert slow_inference_time > fast_inference_time


def test_machine_code():
    v = Value(7.0)
    jv = jit(v)
    assert math.isclose(v.data, jv(), abs_tol=1e-04)

    def float_to_hex(x):
        b = bytearray.fromhex(struct.pack('<f', x).hex())
        b.reverse()
        return b.hex()
    with tempfile.NamedTemporaryFile() as tmp:
        jv.execution_engine.dump_to_object_file(tmp.name)
        result = subprocess.run(
            ['objdump', '-d', '--disassemble=_mlir_net', tmp.name], capture_output=True)
        output = result.stdout.decode('utf-8')
        print(output)
        assert float_to_hex(v.data) in output
