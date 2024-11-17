from flask import Flask, render_template, request

import nltk
from nltk.stem.snowball import GermanStemmer
stemmer = GermanStemmer()
from nltk.corpus import stopwords

import numpy as np
import tflearn
import random
import os
import inspect

def getPath(file):
    path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    path = os.path.join(path, file).replace("\\", "/")
    return path


import pickle
import json



def getJsonPath():
    path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    path = os.path.join(path, 'chat.json').replace("\\", "/")
    return path

# wiederherstelle alle unsere Datenstrukturen
#data = pickle.load(open("./chatbot/trained_data", "rb"))
data = pickle.load(open("./trained_data", "rb"))
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

# importiere die Dialogdesign-Datei
with open(getJsonPath(), encoding="utf8") as json_data:
    dialogflow = json.load(json_data)

# Aufbau des neuronalen Netzes
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 88)
net = tflearn.fully_connected(net, 88)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Definiere das Modell und konfiguriere tensorboard
model = tflearn.DNN(net, tensorboard_dir='train_logs')



app = Flask(__name__)
# run_with_ngrok(app) 

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get", methods=["POST"])
def chatbot_response():
    msg = request.form["msg"]
    res = antwort(msg)
    return res


# chat functionalities
# Bearbeitung der Benutzereingaben, um einen bag-of-words zu erzeugen
def frageBearbeitung(frage):
    # tokenisiere die synonymen
    sentence_word = nltk.word_tokenize(frage)
    # generiere die Stopwörter
    stop = stopwords.words('german')
    ignore_words = ['?', '.', ','] + stop
    ######Korrektur Schreibfehler
    sentence_words = []
    for word in sentence_word:
        if word not in ignore_words or word == 'weiter' or word == 'andere' or word == 'nicht':
            sentence_words.append(word)
    # stemme jedes Wort
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words


# Rückgabe bag of words array: 0 oder 1 für jedes Wort in der 'bag', die im Satz existiert
def bow(frage, words, show_details=False):
    sentence_words = frageBearbeitung(frage)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)

    return (np.array(bag))


# lade unsre gespeicherte Modell
model.load('./model.tflearn')

# Aufbau unseres Antwortprozessors.
# Erstellen einer Datenstruktur, die den Benutzerkontext enthält
context = {}

ERROR_THRESHOLD = 0.10


def klassifizieren(frage):
    # generiere Wahrscheinlichkeiten von dem Modell
    results = model.predict([bow(frage, words)])[0]
    # herausfiltern Vorhersagen unterhalb eines Schwellenwerts
    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
    # nach Stärke der Wahrscheinlichkeit sortieren
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    return return_list


def antwort(frage):
    results = klassifizieren(frage)
    print(results)
    # Wenn wir eine Klassifizierung haben, dann suchen wir das passende dialog-intent
    if results:
        # loop solange es Übereinstimmungen gibt, die verarbeitet werden sollen
        while results:
            for i in dialogflow['dialogflow']:
                # finde ein intent, das dem ersten Ergebnis entspricht
                if i['intent'] == results[0][0]:
                    return random.choice(i['antwort'])
            results.pop(0)


if __name__ == "__main__":
    app.run()

