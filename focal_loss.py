#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf

def focal_loss_softmax(labels,logits,gamma=2):
    """
    Computer focal loss for multi classification
    Args:
      labels: A int32 tensor of shape [batch_size].
      logits: A float32 tensor of shape [batch_size,num_classes].
      gamma: A scalar for focal loss gamma hyper-parameter.
    Returns:
      A tensor of the same shape as `labels`
    """
    epsilon=1.e-8
    y_pred=tf.nn.softmax(logits,axis=-1) # [batch_size,num_classes]
    y=tf.one_hot(labels,depth=y_pred.shape[1])  #output size [batch_size,num_classes]

    L1=-y*((1-y_pred)**gamma)*tf.log(y_pred+epsilon)  # select data with y=1

    L2=-(1-y)*(y_pred**gamma)*tf.log(1-y_pred+epsilon) # select data with y=0

    L=L1+L2

    L=tf.reduce_sum(L, axis=1)
    L=tf.reduce_mean(L,axis=0)
    return L
