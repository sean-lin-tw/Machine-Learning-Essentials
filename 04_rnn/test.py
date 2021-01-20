from tensorflow import keras
from keras.preprocessing import image
from argparse import ArgumentParser
import numpy as np
import gdown
import os

# Parse input arguments
parser = ArgumentParser()
parser.add_argument('file_test_data', help='File path for test data')
parser.add_argument('file_prediction', help='Filename of the prediction results(.csv)')

# Check if model exists; or else download it
model_file = 'hw4.h5'
if not os.path.isfile(model_file):
    url = "https://drive.google.com/u/1/uc?id=1rt0-DRWWNb18Y2RchhAgx3y2ig2fsB7v&export=download"
    gdown.download(url, model_file)

model = keras.models.load_model(model_file)

# Read test data and predict
prediction = []
for filename in os.listdir(dir_test):
    if filename.endswith(".jpg"):
        img_id = int(os.path.splitext(filename)[0])
        img = image.load_img(os.path.join(dir_test, filename), target_size = (128, 128))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis = 0)
        result = np.argmax(model.predict(img))
        prediction.append([img_id, result])

# Write prediction results into a csv file
with open(parser.file_prediction, 'w') as f:
    f.write('id,label\n')
    for img_id, result in prediction:
        f.write('{},{}\n'.format(img_id, result))