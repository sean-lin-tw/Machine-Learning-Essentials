#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import pandas as pd
import numpy as np

# Read csv into table
dataFile = sys.argv[1]
data = pd.read_csv(dataFile, encoding="big5", header=None)
outputFile = open(sys.argv[2] + '.csv', 'w')

attr_select = ['PM2.5']
attr = dict()
for i in range(18):
    attr[data[1][i]] = i

train_hours = int(sys.argv[3])
num_attrs = 18
num_attrs_select = len(attr_select)
num_cols = 9
num_features = num_attrs_select * train_hours + 1 
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
    col = 10 - train_hours + 1
    attr_ctr = 0
    for j in range(num_features):
        row_off = attr[attr_select[attr_ctr]] 
	input_matrix[i][j] = float(data[col][row_base + row_off]) if (j != num_features - 1) else 1.0  
        col = col + 1 if (attr_ctr == num_attrs_select - 1) else col
        attr_ctr = (attr_ctr + 1) % num_attrs_select
    output[i] = np.dot(weight, input_matrix[i])

# Write result to file
outputFile.write('id,value\n')
for i in range(num_inputs):
    outputFile.write('id_' + str(i) + ',' + str(output[i]) + '\n')
outputFile.close()
