import substringFilter as subs
import similarities as sims
import getModel as getmod
import pickle
import sqlite3
import pandas as pd
import getTripleParsContent as threepars
import getNgrams as ngram
import getTokens
import os

li_quarterly_months = ['2017-03','2017-05','2017-06','2017-08','2017-09','2017-11','2017-12','2018-02']
quit()
for filename in os.listdir('tokens/todo'):
	tokens = pickle.load(open('tokens/todo/' + filename, 'rb'))
	getmod.getW2vModel(train=tokens, modelname=filename)
	#pickle.dump(model, open('word_embeddings/word2vec/models_withpars/' + filename ''))

for index, month in enumerate(li_quarterly_months):
	df = subs.substringFilter('all', inmonth=li_quarterly_months[index], tocsv=False)
	li_strings = df['comment'].tolist()
	tokens = getTokens.getTokens(li_strings, lemmatizing=True)
	pickle.dump(tokens, open('tokens/tokens_lemma_withpars_' + month + '.p', 'wb'))
	getmod.getW2vModel(train=tokens, modelname='w2v-withpars-' + month)

# get models
# li_models = ['word_embedding_models/model_withparentheses_2014-01.model','word_embedding_models/model_withparentheses_2015-01.model','word_embedding_models/model_withparentheses_2016-01.model','word_embedding_models/model_withparentheses_2017-01.model','word_embedding_models/model_withparentheses_2018-01.model']

# df = getmod.getWordEmbeddingSimilars('(((they)))', li_models, li_months)
# print(df)
# df.to_csv('w2v_similars_(((they))).csv')