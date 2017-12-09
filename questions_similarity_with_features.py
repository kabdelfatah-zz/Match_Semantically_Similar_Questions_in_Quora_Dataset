
import os
import re
import csv
import codecs
import numpy as np
import pandas as pd
import pydot
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation

from gensim.models import KeyedVectors

os.environ['KERAS_BACKEND']='theano'
from keras.preprocessing.sequence import pad_sequences
from keras.layers import *
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
import sys

import pickle
#====================================================================
MAX_SEQUENCE_LENGTH = 30
MAX_FEATURE_LENGTH = 6
VALIDATION_SPLIT = 0.2
EMBEDDING_DIM = 300
re_weight = True


MAX_NB_WORDS = 100000
print('MAX_NB_WORDS: ',MAX_NB_WORDS)
num_lstm = np.random.randint(175, 275)
num_dense = np.random.randint(100, 150)
rate_drop_lstm = 0.15 + np.random.rand() * 0.25
rate_drop_dense = 0.15 + np.random.rand() * 0.25

act = 'relu'
re_weight = True 

STAMP = 'lstm_%d_%d_%.2f_%.2f'%(num_lstm, num_dense, rate_drop_lstm, \
        rate_drop_dense)

#=============================================================
def split_data( embedding_matrix, Train_features, Test_features):
	Train_features.drop(['id','qid1','qid2','question1','question2'],axis=1,inplace=True)
	Test_features.drop(['question1','question2'],axis=1,inplace=True)

	trainlen = int(len(Train_features)*(1-VALIDATION_SPLIT))
	y_train = Train_features['is_duplicate']
	q1seq_train = Train_features['pdsq1']
	q2seq_train = Train_features['pdsq2']

	test_ids = Test_features['test_id']
	q1seq_test = Test_features['pdsq1']
	q2seq_test = Test_features['pdsq2']

	Train_features.drop(['is_duplicate', 'pdsq1', 'pdsq2', 'sqcq1', 'sqcq2', 'clnq1', 'clnq2'],axis=1,inplace=True)
	Test_features.drop(['test_id', 'pdsq1', 'pdsq2', 'sqcq1', 'sqcq2', 'clnq1', 'clnq2'],axis=1,inplace=True)

	# Split training validation testing
	feature_train = Train_features.head(trainlen)
	label_train = y_train.head(trainlen)
	q1seq_train = q1seq_train.head(trainlen)
	q2seq_train = q2seq_train.head(trainlen)

	feature_val = Train_features.tail(len(Train_features) - trainlen)
	label_val = y_train.head(len(Train_features) - trainlen)
	q1seq_val = q1seq_train.head(len(Train_features) - trainlen)
	q2seq_val = q2seq_train.head(len(Train_features) - trainlen)

	# convert from pandas to np
		
	q1seq_train2 = np.zeros((len(q1seq_train),MAX_SEQUENCE_LENGTH))
	q2seq_train2 = np.zeros((len(q1seq_train),MAX_SEQUENCE_LENGTH))
	q1seq_val2 = np.zeros((len(q1seq_val),MAX_SEQUENCE_LENGTH))
	q2seq_val2 = np.zeros((len(q2seq_val),MAX_SEQUENCE_LENGTH))
	q1seq_test2 = np.zeros((len(q1seq_test),MAX_SEQUENCE_LENGTH))
	q2seq_test2 = np.zeros((len(q2seq_test),MAX_SEQUENCE_LENGTH))

	q1seq_train1 = q1seq_train.map(np.array).as_matrix()
	q2seq_train1 = q2seq_train.map(np.array).as_matrix()
	q1seq_val1 = q1seq_val.map(np.array).as_matrix()
	q2seq_val1 = q2seq_val.map(np.array).as_matrix()
	q1seq_test1 = q1seq_test.map(np.array).as_matrix()
	q2seq_test1 = q2seq_test.map(np.array).as_matrix()

	# q1seq_train2 = q1seq_train.map(np.array)
	# q1seq_train2 = q1seq_train2.as_matrix().reshape(-1,30)
	# print(q1seq_train2.shape)
	# print(q1seq_train2[:10])
	# sys.exit()
	for i in range(len(q1seq_train)):
		q1seq_train2[i,:] = q1seq_train1[i]
		q2seq_train2[i,:] = q2seq_train1[i]

	for i in range(len(q1seq_val2)):
		q1seq_val2[i,:] = q1seq_val1[i]
		q2seq_val2[i,:] = q2seq_val1[i]

	for i in range(len(q1seq_test2)):
		q1seq_test2[i,:] = q1seq_test1[i]
		q2seq_test2[i,:] = q2seq_test1[i]
	
	print(q1seq_test2.shape)
	print(q1seq_test2[:10])
	del q1seq_train
	del q2seq_train
	del q1seq_val
	del q2seq_val
	del q1seq_test
	del q2seq_test

	del q1seq_train1
	del q2seq_train1
	del q1seq_val1
	del q2seq_val1
	del q1seq_test1
	del q2seq_test1

	feature_train = feature_train.as_matrix()
	feature_train = feature_train[:,:-1]
	feature_val = feature_val.as_matrix()
	feature_val = feature_val[:,:-1]
	feature_test = Test_features.as_matrix()
	feature_test = feature_test[:,:-1]

	label_train = label_train.as_matrix()
	label_val = label_val.as_matrix()
	test_ids = test_ids.as_matrix()

	return feature_train, label_train,q1seq_train2,q2seq_train2,  feature_val, label_val, q1seq_val2, q2seq_val2, \
	feature_test, test_ids, q1seq_test2, q2seq_test2, embedding_matrix 


#==============================================================
def LoadPickle(picklpathList):
    pickleObjList = []
    for picklpath in picklpathList:
        print('Loading Pickele...',picklpath)
        with open(picklpath, 'rb') as handle:
            pickleObj = pickle.load(handle)
            print(type(pickleObj))
            # sys.exit()
        handle.close()
        pickleObjList.append(pickleObj)
    return pickleObjList

def create_model_add_features():
    ########################################
    ## define the model structure
    ########################################
    nb_words = embedding_matrix.shape[0]
    embedding_layer = Embedding(nb_words,
            EMBEDDING_DIM,
            weights=[embedding_matrix],
            input_length=MAX_SEQUENCE_LENGTH,
            trainable=False)
    lstm_layer = GRU(100, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)
    #cnn_layer=convolutional.Conv1D(filters=1024, kernel_size=3)
    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    x1 = lstm_layer(embedded_sequences_1)

    #x1=cnn_layer(embedded_sequences_1)

    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    y1 = lstm_layer(embedded_sequences_2)
    #y1 = cnn_layer(embedded_sequences_2)

    #x1=GlobalMaxPooling1D()(x1)
    #y1=GlobalMaxPooling1D()(y1)

    merged = concatenate([x1, y1])
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(num_dense, activation=act)(merged)
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)

    preds_lstm = Dense(1, activation='sigmoid')(merged)
    
    feature_input = Input(shape=(MAX_FEATURE_LENGTH,), dtype='float32')
    f=Dense(10, activation=act)(feature_input)
    preds_f=Dense(1, activation='sigmoid')(f)

    f_merged = concatenate([preds_lstm, preds_f])

    f_merged=Dense(2, activation='sigmoid')(f_merged)
    preds=Dense(1, activation='sigmoid')(f_merged)
    ########################################
    ## train the model
    ########################################
    model = Model(inputs=[sequence_1_input, sequence_2_input,feature_input], \
                  outputs=preds)
    model.compile(loss='binary_crossentropy',optimizer='nadam',metrics=['acc'])
    #model.summary()

    from keras.utils import plot_model
    plot_model(model, to_file='model.png')
    return model



pd_savelist= ['Train_features.pickle','Test_features.pickle', 'embedding_matrix.pickle']

Train_features, Test_features, embedding_matrix = LoadPickle(pd_savelist)
nb_words = embedding_matrix.shape[0]

feature_train, label_train,q1seq_train,q2seq_train,  feature_val, label_val, q1seq_val, q2seq_val, \
	feature_test, test_ids, q1seq_test, q2seq_test,  embedding_matrix = split_data(embedding_matrix, Train_features, Test_features)

del Train_features
del Test_features


weight_val = np.ones(len(label_val))
if re_weight:
    weight_val *= 0.472001959
    weight_val[label_val==0] = 1.309028344 

if re_weight:
    class_weight = {0: 1.309028344, 1: 0.472001959}
else:
    class_weight = None

class_weight = {0: 1.309028344, 1: 0.472001959}

model= create_model_add_features()

print(STAMP)

early_stopping =EarlyStopping(monitor='val_loss', patience=2)
bst_model_path = STAMP + '.h5'
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True)

model.summary()

data_1_train = np.vstack((q1seq_train, q2seq_train))
data_2_train = np.vstack((q2seq_train, q1seq_train))
feature_train= np.vstack((feature_train, feature_train))

label_train = np.concatenate((label_train, label_train))

model.fit([data_1_train, data_2_train, feature_train], label_train, \
        validation_data=([q1seq_val, q2seq_val, feature_val], label_val, weight_val), \
        epochs=100, batch_size=512, shuffle=True, \
        class_weight=class_weight, callbacks=[early_stopping, model_checkpoint])

