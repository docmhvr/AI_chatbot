import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json

with open('intents.json') as file:
    data = json.load(file)

# print(data["intents"])

# Extracting data to lists
words = []
labels = []
docs_x = []
docs_y = []

# Stemming .... loading tags and words data
for intent in data['intents']:
    for pattern in intent['patterns']:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])
        
    if intent['tag'] not in labels:
        labels.append(intent['tag'])

# stemming & words and labels sorted and unique
words = [stemmer.stem(w.lower()) for w in words if w != "?"] # to get to roots 
words = sorted(list(set(words))) # removing duplicates

labels = sorted(labels)

# lists for number of words and labels for model
training = []
output = []

out_empty = [0 for _ in range(len(labels))]

# converting data to feed to model
for x, doc in enumerate(docs_x):
    bag = []
    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag) # [1,2,0,5,4,2,1,...] no. of root words
    output.append(output_row) # [] similar for labels

# converting lists to numpy arrays
training = numpy.array(training)
output = numpy.array(output)

# building the model in tensorflow
tensorflow.reset_default_graph()

# input data
net = tflearn.input_data(shape=[None, len(training[0])])

# 2 hidden layers with 8 neurons
net = tflearn.fully_connected(net, 8) 
net = tflearn.fully_connected(net, 8)

# fully connected layer and regression layer (probablity calculation using activation fucntion)
# output layer with activation function (softmax)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

# training model
model = tflearn.DNN(net) 

# saving model
model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")
