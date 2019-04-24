import numpy as np
from keras.layers import CuDNNLSTM, LSTM, Dense, Embedding, Bidirectional
from keras.utils import plot_model
from keras import Sequential
import matplotlib.pyplot as plt


def model_plot():
    vector_size, max_length, dictionary_size = 300, 65, 120000
    embedding_matrix = np.zeros([dictionary_size, vector_size])
    model = Sequential()
    model.add(Embedding(dictionary_size, vector_size, 
                        weights=[embedding_matrix], 
                        trainable=False, 
                        input_length=max_length))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    plot_model(model, to_file='model.png')


model_plot()
