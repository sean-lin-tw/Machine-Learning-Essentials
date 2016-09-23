#!/usr/bin/python
import sys
from PIL import Image

inputFile = sys.argv[1]
inputImage = Image.open(inputFile)
angle = 180

outputImage = inputImage.rotate(angle)
outputImage.save("ans2.png")
