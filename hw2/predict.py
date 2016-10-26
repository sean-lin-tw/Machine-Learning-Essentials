#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys, argparse
from os.path import isfile
from argparse import ArgumentParser
import pandas as pd
import numpy as np
import math
            
# Program arguments
parser = ArgumentParser(description="Logistic Regression Predictor")
parser.add_argument("-i", "--input", default="spam_data/spam_test.csv", help="test data path")
parser.add_argument("-p", "--parameters", default="parameters.txt", help="test data path")
parser.add_argument("-o", "--output", default="output.csv", help="prediction output")
args = parser.parse_args()  
dataFile, paraFile, outputFile = args.input, args.parameters, args.output

# Read csv into table
data = pd.read_csv(dataFile, header=None, index_col=0)
outputFp = open(outputFile, 'w')

# Normalization
data = (data - data.mean()) / (data.max() - data.min())

# Decide which features to use (include the constant 'bias' term) PM2.5 & PM10
attr = range(1, data.shape[1] + 1)
num_features = len(attr) + 1
num_inputs = data.shape[0] 

weight = np.zeros([num_features])
input_matrix = np.empty([num_inputs, num_features])
output = np.empty([num_inputs])

# Read parameters from file
parameters = pd.read_csv(paraFile, sep='\s+', header=None)
for i in range(num_features):
    weight[i] = parameters[0][i]

# Fill in I/O arrays
for i in range(num_inputs):
    for j in range(num_features):
        input_matrix[i][j] = float(data[attr[j]][i+1]) if (j != num_features - 1) else 1.0    
    #print str(1/(1 + math.exp(-np.dot(weight, input_matrix[i]))))
    output[i] = round(1/(1 + math.exp(-np.dot(weight, input_matrix[i]))))
output = output.astype(int)

# Write result to file
outputFp.write('id,label\n')
for i in range(num_inputs):
    outputFp.write(str(i+1) + ',' + str(output[i]) + '\n')
outputFp.close()
