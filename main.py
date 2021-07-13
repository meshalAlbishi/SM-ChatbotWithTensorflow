from flask import Flask, render_template
import numpy as np
import nltk
import tensorflow as tf
import tflearn
import random
import json
import pickle
from time import sleep

nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer

# import numpy


app = Flask(__name__)
stemmer = LancasterStemmer()

with open('intents.json') as file:
    data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []

for intent in data['intents']:

    for pattern in intent['patterns']:
        # tokenizing the word
        temp_word = nltk.word_tokenize(pattern)
        words.extend(temp_word)
        docs_x.append(temp_word)
        docs_y.append(intent['tag'])

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

    # list contain 0,1 if the word exists
    # if exists put 1
    # if not, put 0
    bag = []

    temp_word = [stemmer.stem(w) for w in doc]

    for w in temp_word:
        # if the word exists append 1, else 0
        bag.append(1 if w in temp_word else 0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    # append the bag and the output row to there lists
    training.append(bag)
    output.append(output_row)

print(training)
# change to numpy arrays, to can work with numpy
training = np.array(training, dtype=object)
output = np.array(output)

# remove any old data of tf in the cache
tf.compat.v1.reset_default_graph()

# start building the model
net = tflearn.input_data(shape=[None, len(training[0])])

# create hidden layers
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation='softmax')
net = tflearn.regression(net)

# start training
model = tflearn.DNN(net)

# n_epoch refer to how many times the model will see the data
model.fit(X_inputs=training, Y_targets=output, n_epoch=1000, batch_size=8, show_metric=True)

# save the model, to not repeat the train
model.save('model.tflearn')


def chat():
    print("Bot: How i can help?")

    while True:
        inp = input("You: ")



@app.route('/')
def index():
    return render_template('pages/home.html')


@app.errorhandler(404)
def not_found_error(error):
    return render_template('errors/404.html'), 404


@app.errorhandler(500)
def server_error(error):
    return render_template('errors/500.html'), 500


if __name__ == '__main__':
    app.run()
