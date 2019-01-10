import mxnet as mx
import horovod.mxnet as hvd
hvd.init()

my_var = mx.symbol.Variable('myconst')
executor = my_var.bind(ctx=mx.cpu(), args={'myconst': mx.nd.array([1, 5])})
executor.forward()
result = hvd.broadcast(executor.outputs[0], root_rank=0)
