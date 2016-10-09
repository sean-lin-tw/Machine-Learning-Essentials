#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import pandas as pd
import numpy as np

# Read csv into table
dataFile = sys.argv[1]
data = pd.read_csv(dataFile, encoding="big5", header=None)
outputFile = open(sys.argv[2] + '.csv', 'w')

train_hours = int(sys.argv[3])
num_attrs = 18
num_cols = 9
num_features = (num_attrs - 1) * train_hours + 1 # skip the 'RAINFALL' row
num_inputs = 240
weight = np.ones([num_features])
input_matrix = np.empty([num_inputs, num_features])
output = np.empty([num_inputs])

# Read parameters from file
parameters = pd.read_csv(sys.argv[4], sep='\s+', header=None)
for i in range(num_features):
    weight[i] = parameters[0][i]

# Fill in I/O arrays
for i in range(num_inputs):
    row_base = 18 * i
    row_off = 0
    col = 10 - train_hours + 1
    for j in range(num_features):
	input_matrix[i][j] = float(data[col][row_base + row_off]) if (j != num_features - 1) else 1.0  
        row_off = 11 if (row_off == 9) else (row_off + 1) % num_attrs
        col = col + 1 if (row_off == 0) else col
    output[i] = np.dot(weight, input_matrix[i])
 
# Write result to file
outputFile.write('id,value\n')
for i in range(num_inputs):
    outputFile.write('id_' + str(i) + ',' + str(output[i]) + '\n')
outputFile.close()
