#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import pandas as pd
import numpy as np

# Read csv into table
filename = sys.argv[1]
data = pd.read_csv(filename, encoding="big5")

# Dimension for each data structure (use numpy to be more space-efficient):
# output: num_inputs
# weight: num_features
# input_matrix: num_inputs * num_features
# gradient: num_features

# Decide which features to use (include the constant 'bias' term) PM2.5 & PM10
train_hours = int(sys.argv[2])
num_attrs = 18
num_attrs_select = 2
num_cols = 24
num_features = (num_attrs_select) * train_hours + 1 # skip the 'RAINFALL' row
num_inputs = 240 * num_cols / (train_hours + 1) 
weight = np.ones([num_features])
input_matrix = np.empty([num_inputs, num_features])
output = np.empty([num_inputs])
gradient = np.empty([num_features])
grad_sqr_sum = np.zeros([num_features])
rate = 10.0

# Fill in I/O arrays
attr_select = ['PM2.5', 'PM10']
attr = dict()
for i in range(18):
    attr[data[u'測項'][i]] = i

row_base = 0
row_off = 0
col = 0
for i in range(num_inputs):
    for j in range(num_features):
	input_matrix[i][j] = float(data[str(col)][row_base + row_off]) if (j != num_features - 1) else 1.0  
	row_base = row_base + num_attrs if (row_off == num_attrs-1 and col == 23) else row_base
        row_off = 11 if (row_off == 9) else (row_off + 1) % num_attrs
        col = (col + 1) % num_cols if (row_off == 0) else col
    output[i] = float(data[str(col)][row_base + 9])
    row_base = row_base + num_attrs if (col == num_cols - 1) else row_base
    row_off = 0
    col = (col + 1) % num_cols

for k in range(1000):
    # Compute gradient descent(with/without regularization)
    for i in range(num_features):
        gradsum = 0.0
        for j in range(num_inputs):
            gradsum = (output[j] - np.dot(weight, input_matrix[j])) * (input_matrix[j][i])
        gradient[i] = gradsum * (-2)
        grad_sqr_sum[i] += gradient[i] ** 2

    # Update the paramaters, using adaptive learning rate (adagrad)
    weight = weight - rate * gradient / (grad_sqr_sum ** 0.5)
    weight = weight - rate * gradient / (grad_sqr_sum ** 0.5)

    # Compute loss function
    loss = 0.0
    for i in range(num_inputs):
        loss += (output[i] - np.dot(weight, input_matrix[i])) ** 2

    # Decide whether to terminate or not(#iterations or error threshold)
'''
    print 'Iteration ' + str(k + 1) + ': \n'
    print 'Weight: \n'
    print weight
    print 'Gradient: \n'
    print gradient
    print 'Grad_sum\n'
    print grad_sqr_sum
    print '\nLoss = ' + str(loss) + '\n'
'''

# write parameters to file
output = open(sys.argv[3], 'w')
for i in range(num_features):
    output.write(str(weight[i]) + '\n')
output.close()
