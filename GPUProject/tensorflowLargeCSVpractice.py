# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 15:26:55 2019

@author: danie
"""
#https://databricks.com/tensorflow/basic-reading-with-python-code

import tensorflow as tf
import os

from os import chdir, getcwd
chdir('C:\\Users\\danie\\Documents\\GitHub\\OlgaDanCapstone\\GPUProject')

dir_path = os.path.dirname(os.path.realpath(__file__))
filename = dir_path + "/olympics2016.csv"

features = tf.placeholder(tf.int32, shape=[3], name='features')
country = tf.placeholder(tf.string, name='country')
total = tf.reduce_sum(features, name='total')