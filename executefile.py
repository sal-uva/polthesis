import substringFilter as subs
import similarities as sims
import pickle
from gensim.models import word2vec

quit()
li_months = ['2014-02', '2015-02', '2016-02', '2017-02', '2018-02','2014-03', '2015-03', '2016-03', '2017-03', '2018-03']
for month in li_months:
	df = subs.substringFilter(inmonth=month)
	#print(df)
	li_posts = df['comment']
	tokens = sims.getTokens(li_strings=li_posts, lemmatizing=True)
	print(tokens[:100])
	pickle.dump(tokens, open('pickle_tokens/' + month + '-all-tokens.p', 'wb'))

# li_months = ['2014-01', '2015-01', '2016-01', '2017-01', '2018-01']
# for month in li_months:
# 	li_strings = pickle.load(open('pickle_tokens/' + month + '-all-tokens.p', 'rb'))
# 	model = sims.getW2vModel(train=li_strings)
# 	print(model.most_similar(positive=['(((they)))']))