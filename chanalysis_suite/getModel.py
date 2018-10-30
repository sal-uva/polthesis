import pandas as pd
from gensim.models import Word2Vec
from gensim.models import fasttext

def getW2vModel(load, train='', modelname='', min_word=200):
	""" Trains or loads a word2vec model. Input must be a list of strings.

	Keyword arguments:
	train -- when provided, trains, saved (in binary) and returns a model
	load -- when provided, loads and returns a model (usually stored in .model.bin)
	modelname -- name of the saved model
	min_word -- the minimum amount of occurances of words to be included in the model. Useful for filtering out bloat.
	
	"""

	if train != '':
		print('Training ' + modelname)
		# train model
		# neighbourhood?
		model = Word2Vec(train, min_count=min_word)
		# pickle the entire model to disk, so we can load&resume training later
		model.save(modelname + '.model')
		#store the learned weights, in a format the original C tool understands
		model.wv.save_word2vec_format(modelname + '.model.bin', binary=True)
		return model
	else:
		model = Word2Vec.load(load)
		return model

def getFastTextModel(train='', load='', modelname='', min_word=200):
	if train != '':
		# train model
		print(train[:10])
		model = fasttext.FastText(sentences=train, min_count=min_word)
		model.save('word_embeddings/fasttext/models/' + modelname + '.model.bin')
		
		# pickle the entire model to load and resume training later
		return model
	elif load != '':
		model = fasttext.FastText.load('word_embeddings/fasttext/models/' + load)
		return model

def getGloveModel(train='', load='', modelname='', min_word=''):
	if train != '':
		# train model
		cooccur = glove.Corpus()
		cooccur.fit(train, window=5)

		# and train GloVe model itself, using 10 epochs
		model_glove = glove.Glove(no_components=100, learning_rate=0.05)
		model_glove.fit(cooccur.matrix, epochs=10)

		model_glove.save('word_embeddings/fasttext/models/' + modelname + '.model.bin')
		
		# pickle the entire model to load and resume training later
		return model
	elif load != '':
		model = fasttext.FastText.load('word_embeddings/fasttext/models/' + load)
		return model

def getWordEmbeddingSimilars(word, li_modelnames, li_months, topn=25, min_word=200):
	"""
	Creates a csv usable for RankFlow with similar terms in a word embedding model
	
	:param	word,			string,	The word to get similarities with.
	:prarm	li_modelnames,	list,	A list of filenames for word embedding models.
									Should be in chronological order.
	:param	li_months,		list,	A list of months the filenames correspond to.
	:param	topn,			int,	Amount of similar words to get.
	"""
	df_similars = pd.DataFrame()

	for index, modelname in enumerate(li_modelnames):
		print(modelname)
		model = getW2vModel(modelname)
		month = li_months[index]
		li_similarwords = []
		li_weights = []

		try:
			similars = model.wv.most_similar(positive=[word], topn = 200)
			total_words = 0
			for words in similars:
				if model.wv.vocab[words[0]].count >= min_word:
					print(model.wv.vocab[words[0]].count)
					li_similarwords.append(words[0])
					li_weights.append(int(words[1] * 100))
					total_words = total_words + 1

					if total_words == topn:
						break
					#df_similars['ratio-' + month] = [model.wv.vocab[words[0]].count for words in similars]
		
		except KeyError:
			df_similars[month] = ['n'] * topn

		df_similars[month] = li_similarwords
		df_similars['ratio-' + month] = li_weights
	return df_similars