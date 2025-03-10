import fasttext

# Пример предварительной обработки текста для фраз
def preprocess_text(text):
    sentences = text.split('\n')
    preprocessed_sentences = ['_'.join(sentence.split()) for sentence in sentences]
    return '\n'.join(preprocessed_sentences)

with open('example-2-phrases.txt', 'w') as f:
    text = """this is an example sentence
              fasttext can learn embeddings
              embeddings are useful for many tasks"""
    f.write(preprocess_text(text))

# Обучение модели на предварительно обработанном корпусе
model = fasttext.train_unsupervised('example-2-phrases.txt', model='skipgram', minCount=1)

# Получение эмбединга для фразы
phrase_vector = model.get_word_vector('example_sentence')
print(phrase_vector)
