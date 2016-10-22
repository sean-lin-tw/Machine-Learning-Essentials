#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import pandas as pd
import numpy as np
import math

# Read csv into table
dataFile = 'spam_data/spam_test.csv'
data = pd.read_csv(dataFile, header=None, index_col=0)
outputFile = open('output.csv', 'w')

# Decide which features to use (include the constant 'bias' term) PM2.5 & PM10
attr = range(1, data.shape[1] + 1)
num_features = len(attr) + 1
num_inputs = data.shape[0] 

train_iterations = 1000
weight = np.zeros([num_features])
input_matrix = np.empty([num_inputs, num_features])
output = np.empty([num_inputs])

# Read parameters from file
parameters = pd.read_csv('parameters.txt', sep='\s+', header=None)
for i in range(num_features):
    weight[i] = parameters[0][i]

# Fill in I/O arrays
for i in range(num_inputs):
    for j in range(num_features):
        input_matrix[i][j] = float(data[attr[j]][i+1]) if (j != num_features - 1) else 1.0    
    print str(1/(1 + math.exp(-np.dot(weight, input_matrix[i]))))
    output[i] = int(round(1/(1 + math.exp(-np.dot(weight, input_matrix[i]))))) 

# Write result to file
outputFile.write('id,label\n')
for i in range(num_inputs):
    outputFile.write(str(i+1) + ',' + str(output[i]) + '\n')
outputFile.close()
