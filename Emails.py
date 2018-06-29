#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 16:32:35 2018

@author: sachin1006

"""
"""
This code consists of training set which has n no. of words for each emailId after extraction.
The rows in training set determines the EmailId's and the columns determine the total words in each emails.

The rows in the test set determines the EmailId's and the columns determine whether the email is a spam or not i.e a ham.
The output will be 0 and 1. 

"""

from __future__ import division
import numpy as np
import sys
import tensorflow as tf
import os
import tarfile  # for extracting the zip file 
import matplotlib.pyplot as plt

# function to apply delimiter and generate text.
def data_extraction_to_csv(filepath,delimiter):
    return np.genfromtxt(filepath,delimiter=delimiter,dtype=None)

#function to extract data
def import_data():
    if 'data' not in os.listdir(os.getcwd()):
        tar_obj=tarfile.open('data.tar.gz')
        tar_obj.extractall()
        tar_obj.close()
        print ('Extracted the data from tarred file')
    else:
        pass
    
    print ('loading training data')
    
    train_X=data_extraction_to_csv('data/trainX.csv',delimiter='\t')
    train_Y=data_extraction_to_csv('data/trainY.csv',delimiter='\t')
    test_X=data_extraction_to_csv('data/testX.csv',delimiter='\t')
    test_Y=data_extraction_to_csv('data/testY.csv',delimiter='\t')
    
    return train_X,train_Y,test_X,test_Y

#Main function
def main(argv):
    train_X,train_Y,test_X,test_Y=import_data()
    
    #Placeholders
    number_of_features=train_X.shape[1]
    number_of_labels=train_Y.shape[1]
    X=tf.placeholder(tf.float,[None,number_of_features])
    Y=tf.placeholder(tf.float,[None,number_of_labels])
    
    #determine the accurate learning rate
    learning_rate=tf.train.exponential_decay(learning_rate=0.01,global_steps=1,decay_steps=0.8)
    
    weight=tf.Variable(tf.random_normal([number_of_features,number_of_labels],stddev=(np.sqrt(6/number_of_labels+number_of_features+1)),name='weights'))
    bias=tf.Variable(tf.random_normal([1,number_of_labels],stddev=(np.sqrt(6/number_of_labels+number_of_features+1)),name='bias'))
    
    sess=tf.InteractiveSession()
    sess.run(global_variables_initializer())
    
    product=tf.matmul(X,weight)+bias
    sigma=tf.sigmoid(product)
    
    drop=tf.placeholder('float32')
    summation=tf.nn.dropout(sum,drop)
    
    
    error=tf.nn.l2_loss(summation-Y,name='squared_error_cost')
    training_step=tf.nn.GradientDescentOptimizer(learning_rate).minimize(error)

    correct_prediction=tf.equal(tf.argmax(Y,1),tf.argmax(summation,1))
    accuracy=tf.reduce_mean(tf.cast(corect_prediction,tf.float32))
    
    
    for i in range(20000):
        
        if(i%100==0):
            train_accuracy=accuracy.eval(feed_dict={X:train_X ,Y:train_Y})
            print ('training step is {} and accuracy is {}'.format(i,train_accuracy))
        sess.run(training_step,feed_dict={X:train_X ,Y:train_Y})
    print ('test accuracy is:{}'.format(accuracy.eval(feed_dict={X:test_X,Y:test_Y})))
        
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_dir',
      type=str,
      default='/tmp/tensorflow/mnist/input_data',
      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
  