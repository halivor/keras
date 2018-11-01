from keras.preprocessing.text import Tokenizer
import numpy as np

samples = ['The cat sat on the mat.', 'The dog ate my homework']
dim = 50
max_length = 10

np.set_printoptions(threshold=np.inf)
res = np.zeros((len(samples), max_length, dim))
for i, s in enumerate(samples):
	for j, w in list(enumerate(s.split()))[:max_length]:
		index = abs(hash(w)) % dim
		res[i, j, index] = 1

print(res)



def simple():
	samples = ['The cat sat on the mat.', 'The dog ate my homework']
	
	tokenizer = Tokenizer(num_words=1000)
	tokenizer.fit_on_texts(samples)
	
	
	sequences = tokenizer.texts_to_sequences(samples)
	print(sequences)
	
	on_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')
	
	word_index = tokenizer.word_index
	print('Found %s unique tokens.' % len(word_index))
