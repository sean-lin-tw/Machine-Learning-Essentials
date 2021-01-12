#!/usr/bin/python
import sys
import pandas as pd
import numpy as np

colNum = int(sys.argv[1])
filename = sys.argv[2]
output = open("ans1.txt", 'w')
data = pd.read_csv(filename, sep='\s+',header=None)
data = data.sort([colNum], ascending=True)
numRow = data.shape[0]
col = data[colNum].tolist()
strCol = map(str, col)
output.write( ', '.join(strCol))
output.close()

