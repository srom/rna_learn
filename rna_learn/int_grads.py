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
import tensorflow as tf


def integrated_gradients_for_binary_features(
    model,
    inputs, 
    baseline,
    target,
):
    """
    Adapted from the original implementation to accomodate binary features.
    """
    inputs_tf = tf.concat((
        tf.convert_to_tensor(baseline, dtype='float32')[tf.newaxis,:], 
        tf.convert_to_tensor(inputs, dtype='float32'),
    ), axis=0)
    target_tf = tf.convert_to_tensor(target, dtype='float32')

    with tf.GradientTape() as t:
        t.watch(inputs_tf)
        y_log_prob = model(inputs_tf).log_prob(target_tf)
        
    grads = t.gradient(y_log_prob, inputs_tf)

    baseline_gradient = grads[0]
    input_gradients = grads[1:]

    int_grads = inputs * (input_gradients + baseline_gradient) / 2

    return tf.reduce_sum(int_grads, axis=2)


def integrated_gradients_for_binary_features_2(
    model,
    inputs, 
    baseline,
    target,
):
    """
    Adapted from the original implementation to accomodate binary features.
    """
    inputs_tf = tf.concat((
        tf.convert_to_tensor(baseline, dtype='float32')[tf.newaxis,:], 
        tf.convert_to_tensor(inputs, dtype='float32'),
    ), axis=0)
    target_tf = tf.convert_to_tensor(target, dtype='float32')

    with tf.GradientTape() as t:
        t.watch(inputs_tf)
        y_log_prob = model(inputs_tf).log_prob(target_tf)
        
    grads = t.gradient(y_log_prob, inputs_tf)

    baseline_gradient = grads[0]
    input_gradients = grads[1:]

    int_grads_baseline = inputs * (input_gradients - baseline_gradient)
    int_grads_baseline_mean =  inputs * (input_gradients + baseline_gradient) / 2
    int_grads_no_baseline = inputs * input_gradients

    return (
        tf.reduce_sum(int_grads_baseline, axis=2),
        tf.reduce_sum(int_grads_baseline_mean, axis=2),
        tf.reduce_sum(int_grads_no_baseline, axis=2),
    )
