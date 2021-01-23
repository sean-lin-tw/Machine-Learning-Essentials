import numpy as np
import pandas as pd
from argparse import ArgumentParser
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential  # NN Activation
from keras.layers import Embedding  # Embedding layer
from keras.layers import Flatten
from keras.layers import Dense  # Fully Connected Networks


def process_command():
    parser = ArgumentParser()
    parser.add_argument('file_training_label', help='Path to the labeled training text data')
    parser.add_argument('file_training_unlabel', help='Path to the unlabeled training text data')

    return parser.parse_args()
    

def word_embedding(docs, vocab_size, max_length):
    encoded_docs = [one_hot(d, vocab_size) for d in docs]
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

    return padded_docs


# returns the model: DNN for classification (1: positive, 0: negative)
def new_model(vocab_size,max_length):
    model = Sequential()
    model.add(Embedding(vocab_size, 8, input_length=max_length))
    model.add(Flatten())
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 
    
    print(model.summary())
    
    return model


def load_training_data(path='data/training_label.txt'):
    if 'training_label' in path:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            sentiment = []
            text = []
            for line in lines:
                parsed = line.split(maxsplit=2)
                sentiment.append(int(parsed[0]))
                text.append(parsed[2].rstrip('\n'))
        
            return pd.DataFrame({'sentiment': sentiment, 'text': text})
    else:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()    
            lines = [line.rstrip('\n') for line in lines]

            return pd.DataFrame({'text': lines})


if __name__ == '__main__':
    '''
    Reading training data
    '''
    args = process_command()
    data_training_label = load_training_data(args.file_training_label)
    data_training_unlabel = load_training_data(args.file_training_unlabel)
    
    '''
    Word Embedding
    vocab_size: integer encode the documents
    max_length: pad documents to a max length of max_length words
    '''
    vocab_size, max_length = 50, 16 
    padded_docs_label = word_embedding(data_training_label['text'], vocab_size, max_length)
    padded_docs_unlabel= word_embedding(data_training_label['text'], vocab_size, max_length)
    
    # TODO: incorpate BOW method

    '''
    Semi-supervised Learning
    '''
    # Use labeled data to train the initial model
    model = new_model(vocab_size, max_length)
    labels = np.array(data_training_label['sentiment'])
    model.fit(padded_docs_label, labels, epochs=50, verbose=0)
    loss, accuracy = model.evaluate(padded_docs_label, labels, verbose=0)
    print('Accuracy: %f' % (accuracy*100))

    # Label the unlabeled training data using the initial model
    new_labels = model.predict(padded_docs_unlabel)
    new_labels[new_labels>=0.5] = 1
    new_labels[new_labels<0.5] = 0
    new_labels = new_labels.reshape(new_labels.shape[0],)

    # Augment the labeled data
    padded_docs_augmented = np.concatenate((padded_docs_label, padded_docs_unlabel), axis=0)
    labels_augmented = np.concatenate((labels, new_labels))

    # Re-train the model using the augmented data
    model = new_model(vocab_size, max_length)
    model.fit(padded_docs_augmented, labels_augmented, epochs=50, verbose=0)
    loss, accuracy = model.evaluate(padded_docs_augmented, labels_augmented, verbose=0)
    print('Accuracy: %f' % (accuracy*100))

    # Saving model
    model.save('hw4.h5')