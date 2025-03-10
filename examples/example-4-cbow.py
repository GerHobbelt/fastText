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

# Тренировка модели FastText с указанными параметрами
model = fasttext.train_unsupervised(
    input='corpus.txt', 
    model='cbow', 
    dim=300, 
    ws=5, 
    epoch=5, 
    minn=5, 
    maxn=5, 
    neg=10,
    t=1e-4,  # Устанавливаем параметр t, который используется для down-sampling частых слов
    thread=4,  # Количество потоков для тренировки
    lr=0.05  # Скорость обучения
)

# Сохранение модели
model.save_model('fasttext_cbow_model.bin')

# Пример получения эмбеддинга для слова
word = 'example'
word_vector = model.get_word_vector(word)
print(f"Vector for word '{word}': {word_vector}")
