import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense
import re
from tensorflow.keras.regularizers import l2

df_train = pd.read_csv('./dataset/twitter_training.csv', header=None)
df_valid = pd.read_csv('./dataset/twitter_validation.csv', header=None)
df_train.columns = ['ID', 'Name', 'Label', 'Content']
df_valid.columns = ['ID', 'Name', 'Label', 'Content']
df_train.dropna(inplace=True)
df_train.drop_duplicates(inplace=True)
df_train['Label'] = df_train['Label'].replace('Irrelevant', 'Neutral')
df_valid['Label'] = df_valid['Label'].replace('Irrelevant', 'Neutral')
label_mapping = {'Positive':0, 'Negative':1, 'Neutral':2}
train_label = df_train['Label'].map(label_mapping)
valid_label = df_valid['Label'].map(label_mapping)

def clean_emoji(tx):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols 
                           u"\U0001F680-\U0001F6FF"  # transport 
                           u"\U0001F1E0-\U0001F1FF"  # flags 
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', tx)

def text_cleaner(tx):
    text = re.sub(r"won\'t", "would not", tx)
    text = re.sub(r"im", "i am", tx)
    text = re.sub(r"Im", "I am", tx)
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"don\'t", "do not", text)
    text = re.sub(r"shouldn\'t", "should not", text)
    text = re.sub(r"needn\'t", "need not", text)
    text = re.sub(r"hasn\'t", "has not", text)
    text = re.sub(r"haven\'t", "have not", text)
    text = re.sub(r"weren\'t", "were not", text)
    text = re.sub(r"mightn\'t", "might not", text)
    text = re.sub(r"didn\'t", "did not", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\!\?\.\@]',' ' , text)
    text = re.sub(r'[!]+' , '!' , text)
    text = re.sub(r'[?]+' , '?' , text)
    text = re.sub(r'[.]+' , '.' , text)
    text = re.sub(r'[@]+' , '@' , text)
    text = re.sub(r'unk' , ' ' , text)
    text = re.sub('\n', '', text)
    text = text.lower()
    text = re.sub(r'[ ]+' , ' ' , text)
    return text

df_train['Content'] = df_train['Content'].apply(lambda x: clean_emoji(x))
df_train['Content'] = df_train['Content'].apply(lambda x: text_cleaner(x))
df_valid['Content'] = df_valid['Content'].apply(lambda x: clean_emoji(x))
df_valid['Content'] = df_valid['Content'].apply(lambda x: text_cleaner(x))

lb = LabelBinarizer()
train_label = lb.fit_transform(train_label)
valid_label = lb.transform(valid_label)

vocabulary_size = 10000
embedding_dim = 64
max_length_sentence = 150
tokenizer = Tokenizer(num_words=vocabulary_size, oov_token='OOV', lower=True, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
tokenizer.fit_on_texts(df_train['Content'])
df_train['Content'] = tokenizer.texts_to_sequences(df_train['Content'])
padding_train_sentences = pad_sequences(df_train['Content'], maxlen= max_length_sentence, truncating= 'post', padding= 'post')
df_valid['Content'] = tokenizer.texts_to_sequences(df_valid['Content'])
padding_valid_sentences = pad_sequences(df_valid['Content'], maxlen= max_length_sentence, truncating= 'post', padding= 'post')


inputs = Input(shape=(max_length_sentence,))  
x = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim)(inputs)
x = Flatten()(x)
x = Dense(10, activation='relu', kernel_regularizer=l2(0.005))(x)
outputs = Dense(3, activation='softmax')(x)
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()
history = model.fit(padding_train_sentences, train_label, epochs=15, batch_size=32, validation_data=(padding_valid_sentences, valid_label))

# Plot training & validation loss
plt.figure(figsize=(8, 5))  # Set figure size
plt.plot(history.history.get('loss', []), label='Train Loss')
plt.plot(history.history.get('val_loss', []), label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()
# Plot training & validation accuracy
plt.figure(figsize=(8, 5))
plt.plot(history.history.get('accuracy', []), label='Train Accuracy')
plt.plot(history.history.get('val_accuracy', []), label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.show()