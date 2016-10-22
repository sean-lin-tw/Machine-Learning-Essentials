#!/usr/bin/python
import sys
import pandas as pd
import numpy as np
import math

# Read csv into table
dataFile = "spam_data/spam_train.csv"
data = pd.read_csv(dataFile, header=None, index_col=0)

# Decide which features to use (include the constant 'bias' term) PM2.5 & PM10
attr = range(1, data.shape[1])
num_features = len(attr) + 1
num_inputs = data.shape[0] 

train_iterations = 10000
weight = np.zeros([num_features])
input_matrix = np.empty([num_inputs, num_features])
output = np.empty([num_inputs])
gradient = np.empty([num_features])
loss_record = [0.0] * train_iterations
reg = 0 
rate = 10e-6
exp_max = math.log(sys.float_info.max)

# Fill in I/O arrays
for i in range(num_inputs):
    for j in range(num_features):
        input_matrix[i][j] = float(data[attr[j]][i+1]) if (j != num_features - 1) else 1.0    
    output[i] = int(data[58][i+1])

# Training process
for k in range(train_iterations):
    # Compute gradient descent(with/without regularization)
    for i in range(num_features):
        gradsum = 0.0
        for j in range(num_inputs):
            z = np.dot(weight, input_matrix[j])
            if z > -exp_max:
                gradsum += (output[j] - 1/(1 + math.exp(-z))) * (input_matrix[j][i])
            else:
                gradsum += output[j] * input_matrix[j][i]
        gradient[i] = (gradsum * (-1) + 2 * reg * weight[i])  / num_inputs

    # Update the paramaters, using adaptive learning rate (adagrad)
    weight = weight - rate * gradient #/ (grad_sqr_sum ** 0.5)                               
    
    # Compute cross entropy   
    entropy = 0.0
    loss = 0
    for i in range(num_inputs):
        z = np.dot(weight, input_matrix[i])
        if z < -exp_max:
            sigma = 0
        else:
            sigma = 1/(1 + math.exp(-z))
        
        if sigma == 0:
            entropy += float(output[i]) * 10e10 + float(1 - output[i]) * math.log(1 - sigma)
        elif sigma == 1:
            entropy += float(output[i]) * math.log(sigma) + float(1 - output[i]) * 10e10
        else:
            entropy += float(output[i]) * math.log(sigma) + float(1 - output[i]) * math.log(1 - sigma)
        
        if int(round(sigma)) != output[i]:
            loss += 1
    entropy = -entropy

    # print statastics
    print 'Iteration ' + str(k + 1) + ': \n'
    print 'Weight: \n'
    print weight
    print 'Gradient: \n'
    print gradient
    print 'Entropy = ' + str(entropy) + '\n' 
    print 'Loss = ' + str(loss) + '\n'

# write parameters to file
output = open('parameters.txt', 'w')
for i in range(num_features):
    output.write(str(weight[i]) + '\n')
output.close()
