import substringFilter as subs
import similarities as sims
import pickle
import sqlite3
import pandas as pd
import getTripleParsContent as threepars
import getNgrams as ngram


li_months = ['2016-05','2016-06','2016-07','2016-08','2016-09','2016-10','2016-11','2016-12','2017-01','2017-02','2017-03','2017-04','2017-05','2017-06','2017-07','2017-08','2017-09','2017-10','2017-11','2017-12','2018-01','2018-02','2018-03']


#df = pd.read_csv('substring_mentions/triplepars_4chan.csv', encoding='utf-8')

ngrams = ngram.getNgrams(querystring='triplepars', fullcomment=False, nsize=2, windowsize = 8, outputlimit = 50, separateontime=True, timeseparator='months')
print(ngrams)


##############################################################

# months = ['2016-05','2016-06','2016-07','2016-08','2016-09','2016-10','2016-11','2016-12','2017-01','2017-02','2017-03','2017-04','2017-05','2017-06','2017-07','2017-08','2017-09','2017-10','2017-11','2017-12','2018-01','2018-02','2018-03']
# df_full = pd.DataFrame()
# for month in months:
# 	df = threepars.getTripleParsContent(inmonth=month, fb=True)
# 	#print(df)

# 	# df = pd.read_csv('substring_mentions/triplepars_' + month + '.csv', encoding='utf-8')
# 	df_month = pd.DataFrame()
# 	df_month['words_' + month] = df.iloc[:,0]
# 	df_month['occurances'] = df.iloc[:,1]
# 	df_full = pd.concat([df_full,df_month], axis=1)
# 	print(df_full[:10])
# df_full.to_csv('allpars.csv', encoding='utf-8', index=False)
# print(df_full)

# df = pd.read_csv('../facebook_seedlist.csv', encoding='utf-8')

# li_fbpages = []
# li_countries = ['CA','AU','US','UK','IE','NZ']

# for page in df.items():
# 	if page['country'] in li_countries:
# 		li_fbpages.append(page['country'])

# print(li_fbpages)

# conn = sqlite3.connect("../fb_scrapelist_alldata.db")
# df = subs.substringFilter(inmonth=month)
# #print(df)
# li_posts = df['comment']
# tokens = sims.getTokens(li_strings=li_posts, lemmatizing=True)