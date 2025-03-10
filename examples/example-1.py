import fasttext

# Пример текстового корпуса
with open('example-1-phrases.txt', 'w') as f:
    f.write("""this is an example sentence
               fasttext can learn embeddings
               embeddings are useful for many tasks""")

# Обучение модели на текстовом корпусе
# Используем параметр `wordNgrams` для указания количества n-грамм
model = fasttext.train_unsupervised('example-1-phrases.txt', model='skipgram', wordNgrams=2, minCount=1)

# Проверка эмбедингов для фраз
phrase = "example sentence"
phrase_vector = sum(model.get_word_vector(word) for word in phrase.split()) / len(phrase.split())
print(phrase_vector)
