#!/bin/bash

wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -P $(dirname $0)
tar -xvzf $(dirname $0)/cifar-10-python.tar.gz -C $(dirname $0)
rm $(dirname $0)/cifar-10-python.tar.gz
