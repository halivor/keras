from keras.datasets import imdb
(tnd, tnl), (ttd, ttl) = imdb.load_data(num_words = 1000)

word_index = imdb.get_word_index()
rwi = dict([(v, k) for (k, v) in word_index.items()])

dr = ' '.join([rwi.get(i - 3, '?') for i in tnd[0]])
print(dr)
