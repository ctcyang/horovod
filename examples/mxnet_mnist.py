# Copyright 2017 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Step 1: import required packages
import argparse
import logging
import mxnet as mx
import horovod.mxnet as hvd

hvd.init()

# CLI
parser = argparse.ArgumentParser(description='MXNet MNIST Example')
parser.add_argument('--batch-size', type=int, default=64,
                    help='training batch size (default: 64)')
parser.add_argument('--dtype', type=str, default='float32',
                    help='training data type (default: float32)')
parser.add_argument('--gpus', type=str, default='0',
                    help='number of gpus to use (default: 0)')
parser.add_argument('--epochs', type=int, default=5,
                    help='number of training epochs (default: 5)')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5,
                    help='SGD momentum (default: 0.5)')
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
logging.info(args)

context = mx.cpu() if args.gpus is None or args.gpus == '0' \
                   else mx.gpu(hvd.local_rank())

# Step 2: data loading
mnist = mx.test_utils.get_mnist()
# Fix the seed
mx.random.seed(42)

batch_size = 100
train_iter = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'],
                               batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'],
                             batch_size)


# Step 3: define network
def mlp():
    # placeholder for data
    data = mx.sym.var('data')
    # Flatten the data from 4-D shape into 2-D 
    # (batch_size, num_channel*width*height)
    data = mx.sym.flatten(data=data)

    # The first fully-connected layer and the corresponding activation function
    fc1 = mx.sym.FullyConnected(data=data, num_hidden=128)
    act1 = mx.sym.Activation(data=fc1, act_type="relu")

    # The second fully-connected layer and the corresponding activation
    # function
    fc2 = mx.sym.FullyConnected(data=act1, num_hidden=64)
    act2 = mx.sym.Activation(data=fc2, act_type="relu")

    # MNIST has 10 classes
    fc3 = mx.sym.FullyConnected(data=act2, num_hidden=10)

    return fc3


# Step 4: fit the model
net = mlp()
# Softmax with cross entropy loss
loss = mx.sym.SoftmaxOutput(data=net, name='softmax')
mlp_model = mx.mod.Module(symbol=loss, context=context)
optimizer_params = {'learning_rate': 0.01}
opt = mx.optimizer.create('sgd', sym=net, **optimizer_params)
opt = hvd.DistributedOptimizer(opt)

mlp_model.fit(train_iter,  # train data
              eval_data=val_iter,  # validation data
              optimizer=opt,  # use SGD to train
              eval_metric='acc',  # report accuracy during training
              batch_end_callback=mx.callback.Speedometer(batch_size),
              num_epoch=args.epochs)  # train for at most 10 dataset passes

# Step 5: Evaluate model accuracy
test_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'],
                              batch_size)
# predict accuracy of mlp
acc = mx.metric.Accuracy()
mlp_model.score(test_iter, acc)
print(acc)
assert acc.get()[1] > 0.94, "Achieved accuracy (%f) is lower than expected \
                            (0.94)" % acc.get()[1]
