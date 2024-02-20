# Neural Net Book

***

This repository follows the tutorial from 
[Neural Networks from Scratch.](https://nnfs.io/)

Has these things from the book:

- [x] Dense Layer [`from ops.layer import LayerDense`](ops/layer.py)
- [x] Activation [`from ops.activation import ...`](ops/activation.py)
  - [x] Linear Activation `ActivationLinear`
  - [x] ReLU Activation `ActivationReLU`
  - [x] Sigmoid Activation `ActivationSigmoid`
  - [x] Softmax Activation `ActivationSoftmax`
    - [x] Softmax Activation with Categorical Cross-entropy loss `ASLCC`
- [x] Loss [`from ops.loss import...`](ops/loss.py)
  - [x] Common Loss Class `Loss`
  - [x] Categorical Cross-entropy loss `LossCatCrossEntropy`
  - [x] Binary Cross-entropy loss `LossBinCrossEntropy`
  - [x] Mean Squared Error loss `LossMSE`
  - [x] Mean Absolute Error loss `LossMAE`
- [x] Accuracy [`from ops.accuracy import...`](ops/accuracy.py)
  - [x] Common Accuracy Class `Accuracy`
  - [x] Regression Accuracy `AccuracyRegression`
  - [x] Categorical Accuracy `AccuracyCategorical`
- [x] Optimizer [`from ops.optimizer import...`](ops/optimizer.py)
  - [x] Gradient Descent Optimizer `OptimizerSDG`
  - [x] Adaptive Gradient Optimizer `OptimizerAdagrad`
  - [x] RMSProp Optimizer `OptimizerRMSProp`
  - [x] Adaptive Momentum Optimizer `OptimizerAdam`

Everything in the scratch folder is just me trying stuff out. 
All the `test_n.py` files are me testing out actual models. 
- [`test_1.py`](test_1.py): Basic Model
- [`test_2.py`](test_2.py): Weight stuff
- [`test_3.py`](test_3.py): Regression model
- [`test_4.py`](test_4.py): First usage of `Model` class with spiral data
- [`test_5.py`](test_5.py): Training model with [mnist fashion database](https://nnfs.io/datasets/fashion_mnist_images.zip)
- [`test_6.py`](test_6.py): Reusing previously trained model


⚠️When importing, **do not do universal import:**

```python
from ops.layer import *  # do not do this !!! !!!
```

this can cause circular imports in some cases.