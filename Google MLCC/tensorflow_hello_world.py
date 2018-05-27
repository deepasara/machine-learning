# -*- coding: utf-8 -*-
"""
Created on Fri May 11 10:05:15 2018

@author: Deepa Manu
"""

import tensorflow as tf

c = tf.constant('Hello, world!')

with tf.Session() as sess:

    print(sess.run(c))