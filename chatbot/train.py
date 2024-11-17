import nltk
from nltk.stem.snowball import GermanStemmer
stemmer = GermanStemmer()

from nltk.corpus import stopwords

import numpy as np
import tensorflow as tf
import tflearn
import random

import os
import inspect

def getPath(file):
    path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    path = os.path.join(path, file).replace("\\", "/")
    return path




import json

def getJsonPath():
    path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    path = os.path.join(path, 'chat.json').replace("\\", "/")
    return path


# importiere das Dialog-design
with open(getJsonPath(), encoding='UTF-8') as json_data:
    dialogflow = json.load(json_data)

words = []
classes = []
documents = []
stop= stopwords.words('german')
ignore_words = ['?', '.', ','] + stop
# loop durch jeden Satz in unseren dialogflow und synonym
for dialog in dialogflow['dialogflow']:
    for pattern in dialog['synonym']:
        # Tokenisieren jedes Wort im Satz
        w = nltk.word_tokenize(pattern)
        # füge die zu unserer Wörterliste hinzu
        words.extend(w)
        # füge die zu Dokumenten in unserem Korpus hinzu
        documents.append((w, dialog['intent']))
        # füge die zu unserer Klassenliste hinzu
        if dialog['intent'] not in classes:
            classes.append(dialog['intent'])

# stemme jedes Word und entferne Duplikate
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words] + ['weit', 'and', 'nicht']
words = sorted(list(set(words)))

# sortiere unsere Klassen
classes = sorted(list(set(classes)))

print(len(documents), "Docs")
print(len(classes), "Classes", classes)
print(len(words), "Split words", words)

# erstelle unsere training data
training = []
output = []
# Erstelle ein leeres Array für unsere Output
output_empty = [0] * len(classes)

# generiere training set und bag of words für jeden Satz
for doc in documents:
    # Initialisierung unsere bag of words
    bag = []
    # Liste der tokenisierte Wörter für den synonym
    pattern_words = doc[0]
    # stemme jedes Wort
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    #  erstelle unsre bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    # output ist '0' für jedes intent und '1' für das aktuelle intent
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# mische unsere Features und verwandle die in np.array
random.shuffle(training)
training = np.array(training)

# Erstelle die Training-Liste
train_x = list(training[:, 0])
train_y = list(training[:, 1])

#tf.compat.v1.reset_default_graph()
# Aufbau des neuronalen Netzes
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 88)
net = tflearn.fully_connected(net, 88)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Definiere das Modell und konfiguriere tensorboard
model = tflearn.DNN(net, tensorboard_dir=getPath('train_logs'))
# Starte das training des Modells
model.fit(train_x, train_y, n_epoch=1000, batch_size=256, show_metric=True)
# Speichere das trainirte Modell
model.save(getPath('model.tflearn'))

print("model created")
#Bearbeitung der Benutzereingaben, um einen bag-of-words zu erzeugen
def frageBearbeitung(frage):
    # tokenisiere die synonymen
    sentence_word = nltk.word_tokenize(frage, language='german')
    # generiere die Stopwörter
    stop= stopwords.words('german')
    ignore_words = ['?', '.', ','] + stop
    ######Korrektur Schreibfehler
    sentence_words=[]
    for word in sentence_word:
        if word not in ignore_words or word=='weiter' or word=='andere' or word=='nicht':
            #a=correction(word)
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




ERROR_THRESHOLD=0
def klassifizieren(frage):
    # generiere Wahrscheinlichkeiten von dem Modell

    p = bow(frage, words, show_details=False)
    results = model.predict(np.array([p]))[0]

        # herausfiltern Vorhersagen unterhalb eines Schwellenwerts
    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
        # nach Stärke der Wahrscheinlichkeit sortieren
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    return return_list
print(klassifizieren('hallo'))



import pickle
# speichere alle unsere Datenstrukturen
pickle.dump({'words': words, 'classes': classes, 'train_x': train_x, 'train_y': train_y},
            open(getPath('trained_data'), "wb"))