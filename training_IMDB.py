"""

Author: Ryan Daniel LeKuch

Date: 4/25/2025

Description: This project will train a neural network to classify
movie reviews from the IMDB dataset as positive or negative using TensorFlow


"""

import pandas as pd
import numpy as np
import os
import gradio as gr

import warnings
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import joblib

# warnings.filterwarnings("ignore")
def setup():
    # Read the CSV file into a pandas dataframe
    df = pd.read_csv("IMDB_Dataset.csv")

    # print(data['sentiment'].head(20))
    # print(data.head(20))

    # print(data.shape) --> (50000, 2)
    # print(type(data)) --> pandas.core.frame.DataFrame

    # print(data.tail()) --> Prints last 5 rows of the dataframe

    # data['sentiment'].value_counts() --> positive: 25000, negative: 25000

    # For easier processing, we will convert the sentiment column to 1 and 0 using one-hot encoding (label encoder)

    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    # data.replace({'sentiment': {'positive': 1, 'negative': 0}}, inplaces-True)

    # print(data.head(20))

    # We will be building our model using LSTM --> Long Short Term Memory

    # Test size is 20% of the data, random_state is used to ensure that the split is reproducible
    # Random_state is 42 becaues it is a common convention in the data science community to use 42 as a random seed
    train_data, test_data = train_test_split(df, test_size = 0.2, random_state = 42)

    # print(train_data.shape) --> (40000, 2)

    # Tokenizer is used to convert the text data into sequences of integers
    # It is how many words should be considered in one sequence, in this case it is 5000 words
    tokenizer = Tokenizer(num_words = 10000)
    tokenizer.fit_on_texts(train_data['review'])

    # Pad_sequences below is used to ensure that all sequences are of the same length
    # The maxlen parameter is used to specify the maximum length of the sequences, converting the words to integers
    X_train = pad_sequences(tokenizer.texts_to_sequences(train_data['review']), maxlen = 200)
    X_test = pad_sequences(tokenizer.texts_to_sequences(test_data['review']), maxlen = 200)

    Y_train = train_data['sentiment']
    Y_test = test_data['sentiment']

    # Building the LSTM model below:
    model = Sequential()

    # Embedding layer is used to convert the input data into dense vectors of fixed size
    # Output_dim is the size of the dense vector, input_length is the length of the input sequences
    model.add(Embedding(input_dim = 10000, output_dim = 128, input_length = 200))
            
    # LSTM layer is used to process the input data, it is a type of recurrent neural network (RNN)
    # The first LSTM layer has 128 units, dropout is used to prevent overfitting, recurrent_dropout is used to prevent overfitting in the recurrent connections
    model.add(LSTM(128, dropout = 0.2, recurrent_dropout = 0.2))

    model.add(Dense(32, activation = 'relu'))
    # Dense layer is used to convert the output of the LSTM layer into a single value (0 or 1)
    # The activation function is sigmoid, which is used for binary classification problems
    model.add(Dense(1, activation = 'sigmoid'))
    
    return model, tokenizer, X_train, Y_train, X_test, Y_test

def train():
    # Load the IMDB dataset
    # The dataset is a CSV file with two columns: review and sentiment
    # The review column contains the text of the review, and the sentiment column contains the sentiment of the review (positive or negative)
    # The dataset is used to train the model to classify movie reviews as positive or negative

    model, tokenizer, X_train, Y_train, X_test, Y_test = setup()

    # Compile the model using the Adam optimizer and binary crossentropy loss function
    # The metrics used to evaluate the model is accuracy
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Batch size is the number of samples used in each iteration of the training process
    # Validation_split is used to split the training data into training and validation data
    model.fit(X_train, Y_train, epochs = 10, batch_size = 64, validation_split = 0.2)

    # if os.path.exists('IMDB_model.h5'):
    #     os.remove('IMDB_model.h5')
    os.makedirs('trained_models', exist_ok=True)
    model.save('trained_models/IMDB_model.h5')
    joblib.dump(tokenizer, 'tokenizer.pkl')


    loss, accuracy = model.evaluate(X_test, Y_test)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")

    # print(loss)
    # print(accuracy)

    # print(model.summary())

# Building The Predictive System
def filter_sequnence(sequence, vocab_size=10000):
    return [[token for token in sequence if token < vocab_size] for sequence in sequences]

def predictive_system(review):

    model = keras.models.load_model('trained_models/IMDB_model.h5')
    tokenizer = joblib.load('tokenizer.pkl')

    sequences = tokenizer.texts_to_sequences([review])
    padded = pad_sequences(sequences, maxlen=200)
    prediction = model.predict(padded)
    sentiment = 'positive' if prediction[0][0] > 0.5 else 'negative'

    if prediction[0][0] > 0.5:
        confidence = prediction[0][0]
    else:
        confidence = 1 - prediction[0][0]

    return sentiment, confidence

def gradio_interface(review):
    sentiment, confidence = predictive_system(review)
    sentiment = str(sentiment).capitalize()
    return f"🧠 Sentiment: {sentiment} \n 🔥 Confidence: {confidence:.2f}"

if __name__ == "__main__":
    # train()

    action = input("Enter 'train' to train the model or 'test' to predict the sentiment of a review: ").strip().lower()

    if action == 'train':
        train()
    elif action == 'test':
        iface = gr.Interface(fn=gradio_interface, inputs="text", outputs="text", title="Movie Review Sentiment Analysis")
        iface.launch()
        # review = input("Enter a movie review (less than 200 characters): ")
        # prediction = predictive_system(review)
        # print(f"The sentiment of the review is: {prediction}")
    else:
        print("Invalid input. Please enter 'train' or 'test'.")



