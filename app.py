import nltk

nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

import numpy
import tensorflow as tf
import tflearn
import random
import json
import pickle

from time import sleep

with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:

        for pattern in intent["patterns"]:
            # tokenizing the word
            temp_word = nltk.word_tokenize(pattern)

            words.extend(temp_word)
            docs_x.append(temp_word)
            docs_y.append(intent["tag"])

            # check if the intent not added before
            # if not, append it to the list
            if intent['tag'] not in labels:
                labels.append(intent['tag'])

    # remove duplicated words
    words = [stemmer.stem(w.lower()) for w in words if w != '?']
    words = sorted(list(set(words)))
    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        # list contain 0 || 1
        # if the word exists put 1
        # if not, put 0
        bag = []

        temp_word = [stemmer.stem(w) for w in doc]

        for w in words:
            bag.append(1 if w in temp_word else 0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        # append the bag and the output row to there lists
        training.append(bag)
        output.append(output_row)

    # change to numpy arrays, to can work with numpy lib
    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)


# remove any old data of tf stored in the cache
tf.compat.v1.reset_default_graph()

# start building the model
net = tflearn.input_data(shape=[None, len(training[0])])

# create hidden layers for the networks
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

# n_epoch refer to how many times the model will see the data
model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")


def bag_of_words(sentence, temp_words):
    # fill the list with 0
    temp_bag = [0 for _ in range(len(temp_words))]

    # s_words is a list of the sentence(s) words
    s_words = nltk.word_tokenize(sentence)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for s in s_words:

        for i, w in enumerate(temp_words):

            # check if the word in the s_words list
            temp_bag[i] = 1 if w == s else 0

    return numpy.array(temp_bag)


def chat():
    print("Hi, How can i help you ?")
    while True:
        user_input = input("You: ")

        if user_input.lower() == "quit" or user_input.lower() == "exit":
            break

        results = model.predict([bag_of_words(user_input, words)])[0]
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        # check the accuracy
        if results[results_index] > 0.8:

            # search for the tag in the intents
            for tg in data["intents"]:

                if tg['tag'] == tag:
                    responses = tg['responses']

            sleep(3)
            bot = random.choice(responses)
            print(bot)

        else:
            print("I don't understand!")


chat()
