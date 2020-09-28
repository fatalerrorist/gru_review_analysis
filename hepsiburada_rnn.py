# -*- coding: utf-8 -*-
"""hepsiburada_rnn.ipynb

Hepsiburada(Turkish Amazon) Comments - Positive and Negative Seperator

We have a dataset for product review comments.

We want to understand is review positive or negative with comments. 

For this case we will use NLP techniques like Tokenizer and Sequence. 

For model building we can use Tensorflow or PyTorch. 

I want use GRU( RNN ) for build layers .
"""

import numpy as np # for building arrays

import pandas as pd # for looking dataset
from tensorflow.python.keras.models import Sequential # for building keras models
from tensorflow.python.keras.layers import Dense, GRU, Embedding , CuDNNGRU # for building model layer ( I will use CuDNNGRU but If you don't have a Nvidia GPU or you don't install CUDA can use classic GRU layer.)
from tensorflow.keras.optimizers import Adam # Adam Optimizer for model optimizing
from tensorflow.python.keras.preprocessing.text import Tokenizer # Tokenizer for text transformation to tokens
from tensorflow.python.keras.preprocessing.sequence import pad_sequences # for fill with 0 to outlier length sentences. (in this case outlier length is 59)


data = pd.read_csv('model_build/hepsiburada.csv')



"""So we have one column data of Rating. 
Rating is emotion  we get from comments.
1 - Positive  0 - Negative 

And other column gives comments.
We will preprocess on comments before building model.

We have too much Positive comments but we don't need make any preprocessing because we have power of Neural Network.
"""

"""
Dependent and Independent values split (X and Y)
"""

Y = data['Rating'].values.tolist()

x = data['Review'].values.tolist()


"""

 We assign cutoff value for train-test split.
"""

cutoff = int(len(x) * 0.80)

"""Train test split with cutoff value (%80)"""

x_train , x_test = x[:cutoff], x[cutoff:]

y_train, y_test = Y[:cutoff], Y[cutoff:]



"""

 Exclude and tokenize less than 10000 repetitive words
"""

num_words = 10000
tokenizer = Tokenizer(num_words=num_words)

tokenizer.fit_on_texts(x)

# tokenizer.word_index    for looking on it

x_train_tokens = tokenizer.texts_to_sequences(x_train)

x_test_tokens = tokenizer.texts_to_sequences(x_test)

"""

RNN algoritmasına bu yorumları vermemiz için hepsinin aynı sayıda kelimeye yani token sayısına (vektörüne) sahip olması gerek

Kelimelerin sayısını bu şekilde bir liste oluşturucusu ile bulabiliriz
"""

num_tokens = [len(tokens) for tokens in x_train_tokens + x_test_tokens]

num_tokens= np.array(num_tokens)


""" Her yorumu aynı sayıda tokene sabitlemek için eşik eder oluşumu - 95 rule- (standart değer*2 + ortalama)"""

max_tokens = int(np.mean(num_tokens)  + 2* np.std(num_tokens))


""" Şimdi ise 59 tokenden az olan kelimeleri 59 tokene kadar 0 sayısıyla dolduran bir fonksiyon kullanarak train ve test atamalarını yapacağız"""

x_train_pad = pad_sequences(x_train_tokens, maxlen= max_tokens)

x_test_pad = pad_sequences(x_test_tokens, maxlen= max_tokens)

""" Gereksiz ve hatalı kelimeleri fonksiyon yazarak atma"""

idx = tokenizer.word_index
inverse_map = dict(zip(idx.values(), idx.keys()))

def tokens_to_string(tokens):
    words = [inverse_map[token] for token in tokens if token!=0]
    text = ' '.join(words)
    return text


model = Sequential()

""" Embedding"""

embedding_size = 50

model.add(Embedding(input_dim= num_words,
                   output_dim= embedding_size,
                   input_length= max_tokens,
                   name= 'embedding_layer'))

""" 3-layer GRU"""

model.add(CuDNNGRU(units=16,return_sequences=True))
model.add(CuDNNGRU(units=8,return_sequences=True))
model.add(CuDNNGRU(units=4))
model.add(Dense(1, activation='sigmoid'))

optimizer = Adam(lr=1e-3)

model.compile(loss='binary_crossentropy',
             optimizer=optimizer,
             metrics=['accuracy'])

model.summary()

x_train_pad = np.array(x_train_pad)

y_train = np.array(y_train)

model.fit(x_train_pad,y_train,epochs=5,batch_size=256)

def process(text_):
    token = Tokenizer.texts_to_sequences(text_)
    tok_pad = pad_sequences(token, maxlen= 59)
    predict  = model.predict(tok_pad)
    return predict[0]




