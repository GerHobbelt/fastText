

## Генерация эмбедингов:

FastText может генерировать эмбединги (векторные представления) для слов и предложений. 
Это одна из основных функций этой библиотеки, разработанной Facebook AI Research (FAIR). 
FastText расширяет возможности Word2Vec, позволяя работать с морфологически сложными языками 
и обрабатывать неизвестные слова с помощью субсловных моделей.

### Для одного слова:

```
word_vector = model.get_word_vector('example')
print(word_vector)
```

### Для целого предложения:

Для предложений FastText не имеет прямого метода, но можно усреднить эмбединги слов в предложении:

```
sentence = "This is an example sentence."
words = sentence.split()
sentence_vector = sum(model.get_word_vector(word) for word in words) / len(words)
print(sentence_vector)
```

Эти шаги помогут вам начать работу с FastText для генерации эмбедингов слов и предложений.

### Обращение к модели FastText через квадратные скобки (model['word']):

Эквивалентно вызову метода get_word_vector('word'). 

Такое использование упрощает код и делает его более читаемым.

```
import fasttext

# Загрузка предобученной модели FastText
model = fasttext.load_model('cc.en.300.bin')s

# Получение вектора слова через метод get_word_vector
word_vector_method = model.get_word_vector('example')

# Получение вектора слова через квадратные скобки
word_vector_brackets = model['example']

# Проверка равенства векторов
print(word_vector_method == word_vector_brackets)
```

### Получение аналогов слова:

Для получения аналогов слова (или близких по смыслу слов) с использованием модели FastText, вы можете использовать метод **get_nearest_neighbors**. Этот метод позволяет найти ближайшие слова к заданному слову на основе косинусного расстояния между их векторными представлениями.

Вот пример, который показывает, как загрузить модель, обучить её на текстовом корпусе, и затем использовать метод **get_nearest_neighbors** для получения аналогов слова:

```
import fasttext

# Пример текстового корпуса
corpus = [
    "This is an example sentence.",
    "FastText can learn embeddings.",
    "Embeddings are useful for many tasks.",
    "Natural language processing is fun."
]

# Запись корпуса в файл
with open('corpus.txt', 'w') as f:
    for sentence in corpus:
        f.write(sentence + '\n')

# Тренировка модели FastText с параметрами по умолчанию (CBOW)
model = fasttext.train_unsupervised(
    input='corpus.txt', model='cbow', dim=300, ws=5, epoch=5, minn=5, maxn=5, neg=10, t=1e-4, thread=4, lr=0.05)

# Сохранение модели
model.save_model('fasttext_cbow_model.bin')

# Пример получения аналогов слова
word = 'example'
nearest_neighbors = model.get_nearest_neighbors(word)

# Печать аналогов слова
print(f"Nearest neighbors for '{word}':")
for neighbor in nearest_neighbors:
    print(f"{neighbor[1]}: {neighbor[0]}")

```

### Работа со словарём

Для извлечения словаря из модели FastText и сохранения его в текстовом виде, вы можете использовать метод get_words(), который возвращает список всех слов, содержащихся в модели. Этот список можно сохранить в файл.

Пример кода, который демонстрирует, как это сделать, если обучаем модель с нуля:

```
import fasttext

# Пример текстового корпуса
corpus = [
    "This is an example sentence.",
    "FastText can learn embeddings.",
    "Embeddings are useful for many tasks.",
    "Natural language processing is fun."
]

# Запись корпуса в файл
with open('corpus.txt', 'w') as f:
    for sentence in corpus:
        f.write(sentence + '\n')

# Тренировка модели FastText с параметрами по умолчанию (CBOW)
model = fasttext.train_unsupervised(
    input='corpus.txt', 
    model='cbow', dim=300, ws=5, epoch=5, minn=5, maxn=5, neg=10, t=1e-4, thread=4, lr=0.05)

# Сохранение модели
model.save_model('fasttext_cbow_model.bin')

# Получение списка всех слов из модели
words = model.get_words()

# Запись словаря в файл
with open('fasttext_vocabulary.txt', 'w') as f:
    for word in words:
        f.write(word + '\n')

print(f"Vocabulary saved to 'fasttext_vocabulary.txt'")
```

### Параметры n-gram

Параметры **minn** и **maxn** в FastText определяют минимальную и максимальную длину символных n-грамм, которые используются при обучении модели. 
Эти n-граммы позволяют модели учитывать части слов и помогают в обучении эмбеддингов для редких или новых слов, которые не встречались в обучающем корпусе.

**Пояснение параметров:**

- **minn:** Минимальная длина символных n-грамм.
- **maxn:** Максимальная длина символных n-грамм.

**Пример:**

Предположим, у нас есть слово "example" и мы установили **minn=3** и **maxn=5**. 
Это значит, что будут использоваться n-граммы длиной от 3 до 5 символов. Вот какие n-граммы будут созданы для слова **"example"**:

**3-граммы:**
- "exa", "xam", "amp", "mpl", "ple"

**4-граммы:**
- "exam", "xamp", "ampl", "mple"

**5-граммы:**
- "examp", "xampl", "ample"

Эти n-граммы используются для обучения модели и помогают ей лучше обрабатывать слова, которые могут не присутствовать целиком в обучающем корпусе, но могут иметь общие подслова с другими словами.

#### Преимущества использования символных n-грамм:

- **Обработка редких слов:** Если слово не встречается в корпусе, но его части (n-граммы) встречаются в других словах, модель может использовать эти части для создания эмбеддинга для нового слова.
- **Работа с морфологией:** Символьные n-граммы помогают учитывать морфологические изменения слов, такие как префиксы и суффиксы.
- **Улучшение обобщающей способности:** Модель может лучше обобщать и создавать более точные эмбеддинги для новых или редких слов.

## Получение эмбедингов для фраз

Если вы хотите, чтобы FastText обучался непосредственно на фразах, вы можете предварительно обработать ваш корпус, чтобы каждая фраза считалась как одно слово, используя специальные символы для обозначения границ фраз. Однако этот метод требует более сложной предварительной обработки данных и изменения формата ввода.

Пример для фразы **example_sentence**:

```
import fasttext

# Пример предварительной обработки текста для фраз
def preprocess_text(text):
    sentences = text.split('\n')
    preprocessed_sentences = ['_'.join(sentence.split()) for sentence in sentences]
    return '\n'.join(preprocessed_sentences)

with open('phrases.txt', 'w') as f:
    text = """this is an example sentence
              fasttext can learn embeddings
              embeddings are useful for many tasks"""
    f.write(preprocess_text(text))

# Обучение модели на предварительно обработанном корпусе
model = fasttext.train_unsupervised('phrases.txt', model='skipgram')

# Получение эмбединга для фразы
phrase_vector = model.get_word_vector('example_sentence')
print(phrase_vector)
```

В этом примере предварительная обработка текста объединяет слова в фразах с помощью подчеркивания **(_)**, 
чтобы FastText воспринимал их как единое слово. Модель обучается на таких "словах-фразах" и может генерировать эмбединги для них.


## Методы для получения представления предложения.

Вот несколько подходов:

- **Обучение модели FastText для фраз**: Вместо того чтобы обучать модель на отдельных словах, можно обучить её на фразах или даже предложениях. Это менее распространённый подход, но он может быть полезен.

- **Использование других моделей для предложений**: Если вашей основной целью является получение эмбедингов для предложений, вы можете рассмотреть использование специализированных моделей, таких как Universal Sentence Encoder, Sentence-BERT, или даже использование трансформеров (например, BERT).

- **Использование архитектур типа RNN или LSTM**: Эти модели хорошо подходят для работы с последовательностями, такими как предложения. Вы можете обучить RNN или LSTM модель для получения эмбедингов предложений.

- **Concatenation или pooling**: Вы можете использовать методы, такие как конкатенация или пулинг (максимальный или минимальный), чтобы объединить эмбединги слов в представление предложения.

Если все же необходимо использовать FastText, то вот пример, как можно получить эмбединги предложений без усреднения:

### Пример с использованием архитектур типа RNN или LSTM для создания эмбедингов предложений (keras)

**example-3-lstm.py**

### Пример с использованием RNN для создания эмбедингов предложений (torch)

Этот пример предполагает использование **PyTorch** для создания модели RNN, которая будет принимать эмбединги слов от FastText и генерировать эмбединги предложений:

```
import fasttext
import torch
import torch.nn as nn

# Загрузка предобученной модели FastText
model = fasttext.load_model('cc.en.300.bin')

# Пример предложения
sentence = "This is an example sentence."

# Получение эмбедингов слов
word_vectors = [model.get_word_vector(word) for word in sentence.split()]
word_vectors = torch.tensor(word_vectors)  # Преобразование в тензор PyTorch

# Определение RNN модели
class SentenceRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(SentenceRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.hidden_size = hidden_size

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size)  # Начальное скрытое состояние
        out, _ = self.rnn(x, h0)
        return out[:, -1, :]  # Последнее скрытое состояние как представление предложения

# Параметры модели
input_size = 300  # Размер эмбедингов FastText
hidden_size = 128  # Размер скрытого состояния RNN

# Создание и использование модели
sentence_rnn = SentenceRNN(input_size, hidden_size)
sentence_embedding = sentence_rnn(word_vectors.unsqueeze(0))  # Добавление размерности батча
print(sentence_embedding)

```


### Наложение TFIDF на эмбединг вместо усреднения

Использование TF-IDF взвешивания на эмбеддингах FastText — интересный подход, который объединяет статистическое взвешивание термов (TF-IDF) и предобученные векторные представления слов (эмбеддинги). 
В этом примере мы покажем, как можно скомбинировать TF-IDF с эмбеддингами FastText для получения векторных представлений предложений.

```
import fasttext
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Загрузка предобученной модели FastText
ft_model = fasttext.load_model('cc.en.300.bin')  # Замените на путь к вашей модели

# Пример данных
sentences = [
    "this is an example sentence",
    "fasttext can learn embeddings",
    "embeddings are useful for many tasks"
]

# Инициализация TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

# Обучение TfidfVectorizer и получение TF-IDF матрицы
tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
feature_names = tfidf_vectorizer.get_feature_names_out()

# Функция для получения взвешенного эмбеддинга предложения
def get_weighted_embedding(sentence, tfidf_vectorizer, tfidf_matrix, ft_model, feature_names):
    # Получение индексов и значений TF-IDF для данного предложения
    response = tfidf_vectorizer.transform([sentence])
    tfidf_scores = {feature_names[i]: response[0, i] for i in range(response.shape[1]) if response[0, i] > 0}
    
    # Инициализация вектора предложения
    weighted_embedding = np.zeros(ft_model.get_dimension())
    
    # Получение взвешенных эмбеддингов
    for word, tfidf_score in tfidf_scores.items():
        if word in ft_model:
            weighted_embedding += ft_model[word] * tfidf_score
            
    # Нормализация вектора предложения
    norm = np.linalg.norm(weighted_embedding)
    if norm > 0:
        weighted_embedding /= norm
    
    return weighted_embedding

# Получение взвешенных эмбеддингов для всех предложений
sentence_embeddings = np.array([get_weighted_embedding(sentence, tfidf_vectorizer, tfidf_matrix, ft_model, feature_names) for sentence in sentences])

# Печать эмбеддингов предложений
print(sentence_embeddings)
```

Комбинирование TF-IDF и эмбеддингов FastText:

**Для каждого предложения:**
- Преобразуем предложение в TF-IDF представление.
- Получаем эмбеддинги слов и умножаем их на соответствующие TF-IDF веса.
- Суммируем взвешенные эмбеддинги слов для получения эмбеддинга предложения.
- Нормализуем итоговый вектор предложения для получения нормализованного эмбеддинга.

Этот пример показывает, как можно скомбинировать TF-IDF взвешивание и предобученные эмбеддинги FastText для получения векторных представлений предложений. Этот метод позволяет учесть важность слов в контексте предложения при создании его эмбеддинга.

## Word vectors for 157 languages

We distribute pre-trained word vectors for 157 languages, trained on Common Crawl and Wikipedia using fastText. 
These models were trained using CBOW with position-weights, in dimension 300, with character n-grams of length 5, a window of size 5 and 10 negatives. 
We also distribute three new word analogy datasets, for French, Hindi and Polish.

## Limitations and bias

Even if the training data used for this model could be characterized as fairly neutral, this model can have biased predictions.

Cosine similarity can be used to measure the similarity between two different word vectors. If two two vectors are identical, the cosine similarity will be 1. For two completely unrelated vectors, the value will be 0. If two vectors have an opposite relationship, the value will be -1.

```
import numpy as np

def cosine_similarity(word1, word2):
    return np.dot(model[word1], model[word2]) / (np.linalg.norm(model[word1]) * np.linalg.norm(model[word2]))

print(cosine_similarity("man", "boy"))

#0.061653383

print(cosine_similarity("man", "ceo"))

#0.11989131

print(cosine_similarity("woman", "ceo"))

#-0.08834904

```
