# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 22:20:13 2020

@author: Owner
"""
import json
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
lemmatizer=WordNetLemmatizer()
words=[]
classes=[]
documents=[]
ignore_words=[]
data_file=open('intents.json').read()
intents=json.loads(data_file)
for intent in intents['intents']:
    for pattern in intent['patterns']:
        w=nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w,intent['tag']))
        #print(w)
        #print(words)
    if(intent['tag'] not in classes):
        classes.append(intent['tag'])
        #print(classes)
words=[lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words=sorted(list(set(words)))
classes=sorted(list(set(classes)))
print(len(documents),"documents",documents)
print(len(classes),"classes",classes)
print(len(words),"unique lemma words",words)
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))
training=[]
output_empty=[0]*len(classes)
for doc in documents:
    bag=[]
    pattern_words=doc[0]
    print(pattern_words)
    pattern_words=[lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    output_row=list(output_empty)
    output_row[classes.index(doc[1])]=1
    print(bag)
    print(output_row)
    training.append([bag,output_row])
random.shuffle(training)
training=np.array(training)
train_x=list(training[:,0])
train_y=list(training[:,1])
print(train_x)
print(train_y)
print("Training data created")
model=Sequential()
model.add(Dense(128,input_shape=(len(train_x[0]),),activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]),activation='softmax'))
sgd=SGD(lr=0.01, decay=1e-6, momentum=0.9)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
modelsave=model.fit(np.array(train_x),np.array(train_y),epochs=200,batch_size=5,verbose=1)
model.save('chatbot.h5',modelsave)
print("model created")



    