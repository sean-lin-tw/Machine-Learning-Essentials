import os
import gdown
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model


def process_command():
    parser = ArgumentParser()
    parser.add_argument('file_test_data', help='File path for test data')
    parser.add_argument('file_prediction', help='Filename of the prediction results(.csv)')

    return parser.parse_args()


def word_embedding(docs, vocab_size, max_length):
    encoded_docs = [one_hot(d, vocab_size) for d in docs]
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

    return padded_docs

def load_testing_data(path='data/testing_data.txt'):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()    
        lines = [line.rstrip('\n') for line in lines]
    
        return pd.DataFrame({'text': lines})


if __name__ == '__main__':
    '''
    Reading testing data
    '''
    args = process_command()
    data_testing = load_testing_data(args.file_test_data)
    vocab_size, max_length = 50, 16 
    padded_docs_testing = word_embedding(data_testing['text'], vocab_size, max_length)
    
    '''
    Loading trained model
    '''
    model_file = 'hw4.h5'
    if not os.path.isfile(model_file):
        url = "https://github.com/jacky6016/Machine-Learning-Practices/releases/download/hw4-rnn-model/hw4.h5"
        gdown.download(url, model_file)

    model = load_model(model_file)

    '''
    Prediction
    '''
    prediction = model.predict(padded_docs_testing)
    prediction = prediction.reshape(prediction.shape[0],)
    prediction[prediction>=0.5] = 1
    prediction[prediction<0.5] = 0
 
    '''
    Write prediction results into a csv file
    '''
    with open(args.file_prediction, 'w') as f:
        f.write('id,label\n')
        for i, y in  enumerate(prediction):
            f.write('{},{}\n'.format(i, int(y)))