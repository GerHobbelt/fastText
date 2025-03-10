
import fasttext
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, Input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Загрузка предобученной модели FastText
ft_model = fasttext.load_model('cc.en.300.bin')

# Пример данных
sentences = [
    "this is an example sentence",
    "fasttext can learn embeddings",
    "embeddings are useful for many tasks"
]

# Параметры
embedding_dim = 300
max_sequence_length = 10

# Получение эмбедингов слов
def get_embedding_matrix(word_index, embedding_dim, ft_model):
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = ft_model[word]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

# Токенизация текста
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
word_index = tokenizer.word_index

# Подготовка данных
data = pad_sequences(sequences, maxlen=max_sequence_length)

# Получение эмбедингов
embedding_matrix = get_embedding_matrix(word_index, embedding_dim, ft_model)

# Определение модели
model = Sequential()
model.add(Embedding(input_dim=len(word_index) + 1,
                    output_dim=embedding_dim,
                    weights=[embedding_matrix],
                    input_length=max_sequence_length,
                    trainable=False))
model.add(LSTM(128, return_sequences=False))
model.add(Dense(64, activation='relu'))
model.add(Dense(embedding_dim))

# Компиляция модели
model.compile(optimizer='adam', loss='mean_squared_error')

# Печать архитектуры модели
model.summary()

# Пример использования модели для получения эмбедингов предложений
sentence_embeddings = model.predict(data)
print(sentence_embeddings)
