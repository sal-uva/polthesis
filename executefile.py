import substringFilter as subs
import similarities as sims
import getModel as getmod
import pickle
import sqlite3
import pandas as pd
import getTripleParsContent as threepars
import getNgrams as ngram


li_models = [
'word_embedding_models/w2v_model_all-06-2015.model',
'word_embedding_models/w2v_model_all-07-2015.model',
'word_embedding_models/w2v_model_all-08-2015.model',
'word_embedding_models/w2v_model_all-09-2015.model',
'word_embedding_models/w2v_model_all-10-2015.model',
'word_embedding_models/w2v_model_all-11-2015.model',
'word_embedding_models/w2v_model_all-12-2015.model',
'word_embedding_models/w2v_model_all-01-2016.model',
'word_embedding_models/w2v_model_all-02-2016.model',
'word_embedding_models/w2v_model_all-03-2016.model',
'word_embedding_models/w2v_model_all-04-2016.model',
'word_embedding_models/w2v_model_all-05-2016.model',
'word_embedding_models/w2v_model_all-06-2016.model',
'word_embedding_models/w2v_model_all-07-2016.model',
'word_embedding_models/w2v_model_all-08-2016.model',
'word_embedding_models/w2v_model_all-09-2016.model',
'word_embedding_models/w2v_model_all-10-2016.model',
'word_embedding_models/w2v_model_all-11-2016.model',
'word_embedding_models/w2v_model_all-12-2016.model',
'word_embedding_models/w2v_model_all-01-2017.model',
'word_embedding_models/w2v_model_all-02-2017.model',
'word_embedding_models/w2v_model_all-03-2017.model',
'word_embedding_models/w2v_model_all-04-2017.model',
'word_embedding_models/w2v_model_all-05-2017.model',
'word_embedding_models/w2v_model_all-06-2017.model',
'word_embedding_models/w2v_model_all-07-2017.model',
'word_embedding_models/w2v_model_all-08-2017.model',
'word_embedding_models/w2v_model_all-09-2017.model',
'word_embedding_models/w2v_model_all-10-2017.model',
'word_embedding_models/w2v_model_all-11-2017.model',
'word_embedding_models/w2v_model_all-12-2017.model',
'word_embedding_models/w2v_model_all-01-2018.model',
'word_embedding_models/w2v_model_all-02-2018.model'
]

li_months = [
'2015-06',
'2015-07',
'2015-08',
'2015-09',
'2015-10',
'2015-11',
'2015-12',
'2016-01',
'2016-02',
'2016-03',
'2016-04',
'2016-05',
'2016-06',
'2016-07',
'2016-08',
'2016-09',
'2016-10',
'2016-11',
'2016-12',
'2017-01',
'2017-02',
'2017-03',
'2017-04',
'2017-05',
'2017-06',
'2017-07',
'2017-08',
'2017-09',
'2017-10',
'2017-11',
'2017-12',
'2018-01',
'2018-02'
]


df = getmod.getWordEmbeddingSimilars('kek', li_models, li_months)
print(df)
df.to_csv('w2v_similars_kek-wordcounts.csv')