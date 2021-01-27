from tensorflow import keras
from keras.preprocessing import image
from argparse import ArgumentParser
import numpy as np
import gdown
import os

# Parse input arguments
parser = ArgumentParser()
parser.add_argument('dir_data', help='Directory path that contains training, validation, and testing data')
parser.add_argument('file_prediction', help='Filename of the prediction results(.csv)')
args = parser.parse_args()
dir_test = os.path.join(args.dir_data, 'testing')

# Check if model exists; or else download it
model_file = 'hw3.h5'
if not os.path.isfile(model_file):
    url = "https://github.com/jacky6016/Machine-Learning-Practices/releases/download/hw3-cnn-model/hw3.h5"
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
with open(args.file_prediction, 'w') as f:
    f.write('Id,Category\n')
    for img_id, result in prediction:
        f.write('{},{}\n'.format(img_id, result))