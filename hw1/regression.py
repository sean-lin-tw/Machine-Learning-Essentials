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
attr_select = ['PM2.5']
train_hours = int(sys.argv[2])
num_attrs = 18
num_attrs_select = len(attr_select)
num_cols = 24
num_features = num_attrs_select * train_hours + 1
num_inputs = 12 * (480 - train_hours) 
weight = np.zeros([num_features])
input_matrix = np.empty([num_inputs, num_features])
output = np.empty([num_inputs])
gradient = np.empty([num_features])
grad_sqr_sum = np.zeros([num_features])
rate = 1.0e-5

# Fill in I/O arrays
attr = dict()
for i in range(18):
    attr[data[u'測項'][i]] = i

row_base = 0
col = 0
for i in range(num_inputs):
    attr_ctr = 0
    for j in range(num_features):
        row_off = attr[attr_select[attr_ctr]] 
	input_matrix[i][j] = float(data[str(col)][row_base + row_off]) if (j != num_features - 1) else 1.0  
	row_base = row_base + num_attrs if (attr_ctr == num_attrs_select - 1 and col == 23 and j != num_features - 1) else row_base
        col = (col + 1) % num_cols if (attr_ctr == num_attrs_select - 1 and j != num_features - 1) else col
        attr_ctr = (attr_ctr + 1) % num_attrs_select 
    output[i] = float(data[str(col)][row_base + 9])
    if (i + 1) % (480 - train_hours) == 0:
        row_base = row_base + num_attrs
        col = 0
    else:
        row_base = row_base - num_attrs if (col < train_hours - 1) else row_base
        col = (col - train_hours + 1 + num_cols) % num_cols
        
for k in range(10000):
    # Compute gradient descent(with/without regularization)
    for i in range(num_features):
        gradsum = 0.0
        for j in range(num_inputs):
            gradsum += (output[j] - np.dot(weight, input_matrix[j])) * (input_matrix[j][i])
        gradient[i] = gradsum * (-2) / num_inputs
        grad_sqr_sum[i] += gradient[i] ** 2

    # Update the paramaters, using adaptive learning rate (adagrad)
    weight = weight - rate * gradient # / (grad_sqr_sum ** 0.5)

    # Compute loss function
    loss = 0.0
    for i in range(num_inputs):
        loss += (output[i] - np.dot(weight, input_matrix[i])) ** 2

    # Decide whether to terminate or not(#iterations or error threshold)
    print 'Iteration ' + str(k + 1) + ': \n'
    print 'Weight: \n'
    print weight
    print 'Gradient: \n'
    print gradient
    print 'Grad_sum\n'
    print grad_sqr_sum
    print '\nLoss = ' + str(loss) + '\n'

# write parameters to file
output = open(sys.argv[3], 'w')
for i in range(num_features):
    output.write(str(weight[i]) + '\n')
output.close()
