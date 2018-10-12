import substringFilter as subs
import similarities as sims
import getModel as getmod
import pickle
import sqlite3
import pandas as pd
import getTripleParsContent as threepars
import getNgrams as ngram
import getTokens

li_quarterly_months = ['2016-04','2016-07','2016-10','2017-01','2017-04','2017-07','2017-10','2018-01']

for index, month in enumerate(li_quarterly_months):
	df = subs.substringFilter('all', inmonth=li_quarterly_months[index], tocsv=False)
	li_strings = df['comment'].tolist()
	tokens = getTokens.getTokens(li_strings, lemmatizing=True)
	pickle.dump(tokens, open('tokens/tokens_lemma_withpars_' + month + '.p', 'wb'))

# get models
# li_models = ['word_embedding_models/model_withparentheses_2014-01.model','word_embedding_models/model_withparentheses_2015-01.model','word_embedding_models/model_withparentheses_2016-01.model','word_embedding_models/model_withparentheses_2017-01.model','word_embedding_models/model_withparentheses_2018-01.model']

# df = getmod.getWordEmbeddingSimilars('(((they)))', li_models, li_months)
# print(df)
# df.to_csv('w2v_similars_(((they))).csv')