import pandas as pd
df=pd.read_csv('DGA-Alexa2022-label.csv')
print(df.head())
X=df.drop('Label',axis=1)
y=df['Label']
print(X.shape,y.shape)
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense

### Vocabulary size
voc_size=1000000

urls=X.copy()
print(urls['URL'][0])
urls.reset_index(inplace=True)
### Dataset Preprocessing
corpus = []
for i in range(0, len(urls)):
    #print(i)
    url = re.sub('[^a-zA-Z0-9]', ' ', urls['URL'][i])
    url = url.lower()
    url = url.split()
    url = ' '.join(url)
    corpus.append(url)

onehot_repr=[one_hot(words,voc_size)for words in corpus] 
onehot_repr

sent_length=5
embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
print(embedded_docs)

## Creating model
embedding_vector_features=40
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
model.add(LSTM(100))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())


print(len(embedded_docs),y.shape)

import numpy as np
X_final=np.array(embedded_docs)
y_final=np.array(y)


print(X_final.shape,y_final.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.20, random_state=42)

### Finally Training
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=64)


from tensorflow.keras.layers import Dropout
## Creating model
embedding_vector_features=40
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
model.add(Dropout(0.3))
model.add(LSTM(100))
model.add(Dropout(0.3))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

from sklearn.metrics import confusion_matrix

y_pred=model.predict_classes(X_test)

print(confusion_matrix(y_test,y_pred))
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))
