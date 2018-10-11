import substringFilter as subs
import similarities as sims
import pickle
import sqlite3
import pandas as pd
import getTripleParsContent as threepars
import getNgrams as ngram

conn = sqlite3.connect("C:/Users/hagen/Documents/UvA/datasprints/may_2018_springsprint/fb_scrapelist_alldata.db")
df = pd.read_sql_query("SELECT * FROM fbpagedata WHERE lower(message) LIKE '%take your liberal butt%';", conn)
print(df)
df.to_csv('test.csv',encoding='utf-8')