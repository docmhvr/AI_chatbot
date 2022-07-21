import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle

with open('intents.json') as file:
    data = json.load(file)

# If preprocessed data present no need to do again
try:
    with open("data.pickle", "rb") as f:
        words,labels,training,output = pickle.load(f)

except:
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

    # saving model to pickle file
    with open("data.pickle", "wb") as f:
        pickle.dump((words,labels,training,output), f)

# building the model in tensorflow
tensorflow.compat.v1.reset_default_graph()

# input data
net = tflearn.input_data(shape=[None, len(training[0])])

# 2 hidden layers with 8 neurons
net = tflearn.fully_connected(net, 8) 
net = tflearn.fully_connected(net, 8)

# fully connected layer and regression layer (probablity calculation using activation fucntion)
# output layer with activation function (softmax)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

# training model DNN is a type of neural network
model = tflearn.DNN(net) 

# training and saving the DNN model
try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

#The bag_of_words function will transform our string input to a bag of words using our created words list. 
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)

# The chat() getting a prediction from the model and grabbing an appropriate response.
def chat():
    print("You are now talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        print(random.choice(responses))

chat()