"""
Integrated gradients

Integrated gradients is a technique for attributing a deep network's
prediction to its input features.

Introduced in "Axiomatic Attribution for Deep Networks,
Mukund Sundararajan, Ankur Taly, Qiqi Yan
Proceedings of International Conference on Machine Learning (ICML), 2017"

Paper:
http://proceedings.mlr.press/v70/sundararajan17a/sundararajan17a.pdf

Original codebase: 
https://github.com/ankurtaly/Integrated-Gradients

Goal:
Gradient is commonly used as a window into the inner workings of a deep NN. 
However, as shown in the paper, gradient alone suffers from a number
of issue when it comes to interpretability.
Integrated gradients alleviate these shortcomings.

For example, gradient violates the axiom of sensitivity, which is defined as follow:
If for every input and baseline that differ in one feature but have different 
predictions then the differing feature should be given a non-zero attribution.

Basic intuition:
Gradient with respect to the input is compared against 
a baseline, e.g. the zero vector.

Method:
Integrated gradients are obtained by computing the path integral of the 
gradients along the straight line in R^n from the baseline to the input.

See paper for more details.
"""
import numpy as np
import tensorflow as tf


def integrated_gradients_for_binary_features(
    inputs, 
    baseline,
    target,
    gradient_fn,
):
    """
    Adapted from the original implementation to accomodate binary features.
    """
    inp = np.vstack((baseline[np.newaxis,:], inputs))
    grads = gradient_fn(inp, target)

    baseline_gradient = grads[0]
    input_gradients = grads[1:]

    int_grads = inputs * (input_gradients - baseline_gradient)

    return np.sum(int_grads, axis=2)


def make_gradient_fn(model):

    def gradient_fn(inputs, target):
        with tf.GradientTape() as t:
            inputs_tf = tf.convert_to_tensor(inputs, dtype='float32')
            t.watch(inputs_tf)
            y_log_prob = model(inputs_tf).log_prob(target)
            
        return t.gradient(y_log_prob, inputs_tf).numpy()

    return gradient_fn
