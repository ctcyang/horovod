import mxnet as mx
import horovod.mxnet as hvd
hvd.init()
my_var = mx.ndarray.array([1,5])
result = hvd.broadcast(my_var, root_rank=0)
