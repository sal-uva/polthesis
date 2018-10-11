import substringFilter as subs
import similarities as sims
import getModel as getmod
import pickle
import sqlite3
import pandas as pd
import getTripleParsContent as threepars
import getNgrams as ngram

li_models = ['word_embedding_models/model_withparentheses_2014-01.model','word_embedding_models/model_withparentheses_2015-01.model','word_embedding_models/model_withparentheses_2016-01.model','word_embedding_models/model_withparentheses_2017-01.model','word_embedding_models/model_withparentheses_2018-01.model']
li_months = ['2014-01','2015-01','2016-01','2017-01','2018-01']

df = getmod.getWordEmbeddingSimilars('(((they)))', li_models, li_months)
print(df)
df.to_csv('w2v_similars_(((they))).csv')

# for model_name in li_models:
# 	model = getmod.getW2vModel('word_embedding_models/' + model_name)
# 	print(model.most_similar(positive=['kek']))