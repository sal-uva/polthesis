from __future__ import print_function
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt, mpld3
import time
import re
import os
import nltk
import pickle
from matplotlib.font_manager import FontProperties
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from scipy.interpolate import spline
from datetime import datetime, timedelta
from collections import OrderedDict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
from gensim.scripts.word2vec2tensor import word2vec2tensor
from matplotlib import pyplot
from adjustText import adjust_text

#month variables, so I don't have to mess with datetime
li_months = ['10-2015','11-2015','12-2015','01-2016','02-2016','03-2016','04-2016','05-2016','06-2016','07-2016','08-2016','09-2016','10-2016','11-2016','12-2016','01-2017','02-2017','03-2017','04-2017','05-2017','06-2017','07-2017','08-2017','09-2017','10-2017','11-2017','12-2017','01-2018','02-2018','03-2018']

li_filenames = ['01-16.csv', '01-17.csv', '01-18.csv', '02-16.csv', '02-17.csv', '02-18.csv', '03-16.csv', '03-17.csv', '04-16.csv', '04-17.csv', '05-16.csv', '05-17.csv', '06-15.csv', '06-16.csv', '06-17.csv', '07-15.csv', '07-16.csv', '07-17.csv', '08-15.csv', '08-16.csv', '08-17.csv', '09-15.csv', '09-16.csv', '09-17.csv', '10-15.csv', '10-16.csv', '10-17.csv', '11-15.csv', '11-16.csv', '11-17.csv', '12-15.csv', '12-16.csv', '12-17.csv']

li_labels_months = ['06-15', '07-15', '08-15', '09-15', '10-15', '11-15', '12-15', '01-16', '02-16', '03-16', '04-16', '05-16', '06-16', '07-16', '08-16', '09-16', '10-16', '11-16', '12-16', '01-17', '02-17', '03-17', '04-17', '05-17', '06-17', '07-17', '08-17', '09-17', '10-17', '11-17', '12-17','01-18', '02-18']

# di_stems = {}
# di_stems['randomwoord1212'] = 0
# pickle.dump(di_stems, open('di_stems.p', 'wb'))
# quit()

def getTokens(li_strings='', stemming=False):
	if stemming:
		global di_stems
		di_stems = pickle.load(open('di_stems.p', 'rb'))

	print('imported')
	#do some cleanup: only alphabetic characters, no stopwords
	# create separate stemmed tokens, to which the full strings will be compared to:
	li_comments_stemmed = []
	len_comments = len(li_strings)
	print(len(li_strings))
	print('Creating list of tokens per monthly document')
	for index, comment in enumerate(li_strings):
		#create list of list for comments and tokens
		if isinstance(comment, str):
			li_comment_stemmed = []
			li_comment_stemmed = tokeniserAndStemmer(comment, stemming=stemming)
			li_comments_stemmed.append(li_comment_stemmed)
		if index % 1000 == 0:
			print('Stemming/tokenising finished for string ' + str(index) + '/' + str(len_comments))
	print(len(li_comments_stemmed))

	if stemming:
		pickle.dump(di_stems, open('di_stems.p', 'wb'))
		df_stems = pd.DataFrame.from_dict(di_stems, orient='index')
		df_stems.to_csv('di_stems_dataframe.csv', encoding='utf-8')

	return li_comments_stemmed

def tokeniserAndStemmer(string, stemming=False):
	#first, remove urls
	if 'http' in string:
		string = re.sub(r'https?:\/\/.*[\r\n]*', ' ', string)
	if 'www.' in string:
		string = re.sub(r'www.*[\r\n]*', ' ', string)

	#use nltk's tokeniser to get a list of words
	tokens = [word for sent in nltk.sent_tokenize(string) for word in nltk.word_tokenize(sent)]
	stemmer = SnowballStemmer("english")
	#list with tokens further processed
	li_filtered_tokens = []
	# filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
	for token in tokens:
		#only alphabetic characters
		if re.search('[a-zA-Z]', token):
			#only tokens with three or more characters
			if len(token) >= 3:
				#no stopwords
				if token not in stopwords.words('english'):
					token = token.lower()
					#shorten word if it's longer than 20 characters (e.g. 'reeeeeeeeeeeeeeeeeeeeeeeee')
					if len(token) >= 20:
						token = token[:20]
					#stem if indicated it should be stemmed
					if stemming:
						token_stemmed = stemmer.stem(token)
						li_filtered_tokens.append(token_stemmed)

						#update lookup dict with token and stemmed token
						#lookup dict is dict of stemmed words as keys and lists as full tokens
						if token_stemmed in di_stems:
							if token not in di_stems[token_stemmed]:
								di_stems[token_stemmed].append(token)
						else:
							di_stems[token_stemmed] = []
							di_stems[token_stemmed].append(token)
					else:
						li_filtered_tokens.append(token)
	return li_filtered_tokens

def getDocSimilarity(li_strings='', dates='', querystring='', load=False, kmeansgraph=False):
	# look up iterative k-means
	if load == False:
		#max_df used to filter out words like 'like' and 'trump'. Check https://stackoverflow.com/questions/46118910/scikit-learn-vectorizer-max-features?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
		tfidf_vectorizer = TfidfVectorizer(min_df=3, max_df=32, stop_words='english', analyzer='word', token_pattern=u'(?u)[a-zA-Z]{3,}')
		print('Creating tf-idf vector of input documents')
		#prepare vectorizer
		#create tf_idf vectors of month-separated comments
		tfidf_matrix = tfidf_vectorizer.fit_transform(li_strings)
		pickle.dump(tfidf_vectorizer, open('tfidf/trump_tfidf_vectorizer.p', 'wb'))
		pickle.dump(tfidf_matrix, open('tfidf/trump_tfidf_matrix.p', 'wb'))
		pickle.dump(tfidf_matrix, open('tfidf/trump_li_strings.p', 'wb'))
	else:
		tfidf_vectorizer = pickle.load(open('tfidf/trump_tfidf_vectorizer.p', 'rb'))
		tfidf_matrix = pickle.load(open('tfidf/trump_tfidf_matrix.p', 'rb'))
		li_strings = pickle.load(open('tfidf/trump_li_strings.p', 'rb'))

	print(tfidf_matrix[:10])

	feature_array = np.array(tfidf_vectorizer.get_feature_names())
	tfidf_sorting = np.argsort(tfidf_matrix.toarray()).flatten()[::-1]

	n = 100
	top_n = feature_array[tfidf_sorting][:n]
	print(top_n)

	weights = np.asarray(tfidf_matrix.mean(axis=0)).ravel().tolist()
	df_weights = pd.DataFrame({'term': tfidf_vectorizer.get_feature_names(), 'weight': weights})
	df_weights.sort_values(by='weight', ascending=False).head(20)
	df_weights.to_csv('tfidf/top25words.csv', encoding='utf-8')

	df_matrix = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names())
	#turn the csv 90 degrees
	df_matrix = df_matrix.transpose()
	print('Amount of words: ' + str(len(df_matrix)))
	#do some 
	df_matrix.columns = li_filenames
	cols = df_matrix.columns.tolist()
	cols = ['06-15.csv', '07-15.csv', '08-15.csv', '09-15.csv', '10-15.csv', '11-15.csv', '12-15.csv', '01-16.csv', '02-16.csv','03-16.csv','04-16.csv','05-16.csv','06-16.csv','07-16.csv','08-16.csv','09-16.csv','10-16.csv','11-16.csv','12-16.csv', '01-17.csv', '02-17.csv', '03-17.csv', '04-17.csv', '05-17.csv', '06-17.csv', '07-17.csv', '08-17.csv', '09-17.csv', '10-17.csv', '11-17.csv', '12-17.csv', '01-18.csv', '02-18.csv']
	df_matrix = df_matrix[cols]
	df_matrix.to_csv('tfidf/trump_matrix.csv', encoding='utf-8')
	
	cosine_sim = (tfidf_matrix * tfidf_matrix.T).toarray()
	print(cosine_sim)
	df_cosine_matrix = pd.DataFrame(cosine_sim)
	df_cosine_matrix.to_csv('tfidf/trump_cosine_matrix.csv', encoding='utf-8')

	frames = []
	for month in df_matrix:
		print(month)
		df_test = df_matrix.sort_values(by=[month], ascending=False)
		frames.append(df_test[:50])
		print(df_test[month][:10])
	df_top_terms = pd.concat(frames, axis=1)
	df_top_terms.to_csv('tfidf/topterms.csv', encoding='utf-8')

	if kmeansgraph:
		print('Calculating document similarities')
		terms = tfidf_vectorizer.get_feature_names()
		#print(terms)
		dist = 1 - cosine_similarity(tfidf_matrix)
		print(dist)

		num_clusters = 7

		# create new K-means clusters
		k_means = KMeans(n_clusters = num_clusters)
		k_means.fit(tfidf_matrix)
		clusters = k_means.labels_.tolist()
		print(clusters)
		joblib.dump(k_means, 'clusters/doc_cluster_' + querystring + '_ngram.pkl')

		# loading existing clusters for debugging/testing
		# k_means = joblib.load('doc_cluster.pkl')
		# clusters = k_means.labels_.tolist()
		di_clusters = {'dates': dates, 'text': li_strings, 'cluster': clusters}

		df_kclusters = pd.DataFrame(di_clusters, index=[clusters], columns = ['dates', 'cluster'])

		df_kclusters.to_csv('clusters/cluster_'+ querystring + '_ngram.csv')

		#sort cluster centers by proximity to centroid
		order_centroids = k_means.cluster_centers_.argsort()[:, ::-1] 

		di_cluster_names = {}
		for i in range(num_clusters):
			clusterstring = ''
			print("Cluster %d:" % i),
			for ind in order_centroids[i, :15]:
				print(' %s' % terms[ind])
				clusterstring += ' ' + terms[ind]
			di_cluster_names[i] = clusterstring
		
		di_cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e', 5: '#b75d5d', 6: '#e0d100'}
		MDS()

		# convert two components as we're plotting points in a two-dimensional plane
		# "precomputed" because we provide a distance matrix
		# we will also specify 'random_state' so the plot is reproducible.
		mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

		pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

		xs, ys = pos[:, 0], pos[:, 1]
		print()
		print()

		#create data frame that has the result of the MDS plus the cluster numbers and titles
		df_plot = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=dates)) 

		#group by cluster
		groups = df_plot.groupby('label')

		fig, ax = plt.subplots(figsize=(10, 8)) # set size
		ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

		#iterate through groups to layer the plot
		#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
		for name, group in groups:
		    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, 
		            label=di_cluster_names[name], color=di_cluster_colors[name], 
		            mec='none')
		    ax.set_aspect('auto')
		    ax.tick_params(\
		        axis= 'x',          # changes apply to the x-axis
		        which='both',      # both major and minor ticks are affected
		        bottom='off',      # ticks along the bottom edge are off
		        top='off',         # ticks along the top edge are off
		        labelbottom='off')
		    ax.tick_params(\
		        axis= 'y',         # changes apply to the y-axis
		        which='both',      # both major and minor ticks are affected
		        left='off',      # ticks along the bottom edge are off
		        top='off',         # ticks along the top edge are off
		        labelleft='off')

		#add label in x,y position with the label as the date
		for i in range(len(df_plot)):
			ax.text(df_plot.ix[i]['x'], df_plot.ix[i]['y'], df_plot.ix[i]['title'], size=8)  
		
		fontP = FontProperties()
		fontP.set_size('small')
		# Shrink current axis's height by 10% on the bottom
		box = ax.get_position()

		ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
		# Put a legend below current axis
		legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), borderaxespad=0., fancybox=True, shadow=True, ncol=1, prop = fontP)

		plt.setp(legend.get_title(), fontsize='small')
		plt.title('K-means clusters of time-separated documents, comments containing "' + querystring + '"')

		plt.show() #show the plot

		#uncomment the below to save the plot if need be
		plt.savefig('clusters_small_noaxes.png', dpi=200)

def getWord2VecModel(train='', load='', modelname='', min_word=200):
	if train != '':
		# train model
		# neighbourhood?
		model = Word2Vec(train, min_count=min_word)
		# pickle the entire model to disk, so we can load&resume training later
		model.save('word2vec/models/' + modelname + '.model')
		#store the learned weights, in a format the original C tool understands
		model.wv.save_word2vec_format('word2vec/models/' + modelname + '.model.bin', binary=True)
		return model
	elif load != '':
		model = Word2Vec.load(load)
		return model
	
def showPCAGraph(model):
	# use t-sne!
	# PCA is more effective for 'importance' of words

	# fit a 2d PCA model to the vectors
	X = model[model.wv.vocab]
	pca = PCA(n_components=80)
	result = pca.fit_transform(X)
	# create a scatter plot of the projection

	pyplot.scatter(result[:, 0], result[:, 1])
	words = list(model.wv.vocab)
	for i, word in enumerate(words):
		pyplot.annotate(word, xy=(result[i, 0], result[i, 1]), size=6)
	plt.rcParams.update({'font.size': 3})
	pyplot.show()

# some calls for these function come from substring
def getSimilaritiesFromCsv(df, modelname = ''):
	#df = pd.read_csv(csvdoc, encoding='utf-8')
	li_strings = []
	for comment in df['comment']:
		li_strings.append(comment)
	words_stemmed = getTokens(li_strings, similaritytype='words', stems=False)
	#print(words_stemmed[:100])
	#df_stemmedwords = pd.DataFrame(words_stemmed)

	pickle.dump(words_stemmed, open("word2vec/pickle_stems/pickle_" + modelname + ".p", "wb"))
	model = getWord2VecModel(train=words_stemmed, modelname=modelname)
	# model = getWord2VecModel(load=modelname)
	#showPCAGraph(model)
	# similars = model.most_similar(positive=['btfo'], topn = 20)
	# print(similars)
	# similars = model.similar_by_vector(model['hillari'] + model['polit'])
	# print(similars)

def getTsneScatterPlot(model, plotname='', perplexity=10):
	print('getting vocab')
	vocab = list(model.wv.vocab)
	X = model[vocab]
	#TSNE args: perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23
	tsne = TSNE(n_components=2, perplexity=perplexity)
	print('fitting TSNE')
	X_tsne = tsne.fit_transform(X)
	print('writing DataFrame')
	df = pd.DataFrame(X_tsne, index=vocab, columns=['x', 'y'])
	print('creating plt figure')
	fig = plt.figure(figsize=(15, 13))
	ax = fig.add_subplot(1, 1, 1)

	scatter = ax.scatter(df['x'].tolist(), df['y'].tolist(), facecolors='none', edgecolors='none')
	labels = []
	for word, pos in df.iterrows():
		if 'trump' in word:
			ax.annotate(word, pos, fontsize=17, color='#E1313199')
			ax.set_zorder(1000)
		if 'haha' in word or 'lol' in word or 'reee' in word or 'lmfao' in word:
			ax.annotate(word, pos, fontsize=17, color='#3F902790')
			ax.set_zorder(1000)
		else:
			ax.annotate(word, pos, fontsize=7, color='#4f4f4f80')
			ax.set_zorder(10)
		labels.append(word)
	#adjust_text(labels, force_text=0.05, arrowprops=dict(arrowstyle="-|>", color='gray', alpha=0.1))

	#save the mpl figures to pickle and zoom in later
	pickle.dump(fig, open(r'word2vec/tsne/mpl_tsnescatterplot_' + plotname + '.pickle', 'wb'))
	
	css='*{font-family: Arial, sans-serif;}'
	tooltip2 = mpld3.plugins.PointHTMLTooltip(fig, css=css)
	mpld3.plugins.connect(fig, tooltip2)
	#add interactive labels
	tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=labels)
	mpld3.plugins.connect(fig, tooltip)
	#mpld3.show()
	#save to html
	mpld3.save_html(fig, 'word2vec/tsne/mpl_tsnescatterplot_' + plotname + '.html')
	plt.savefig('word2vec/tsne/mpl_tsnescatterplot_' + plotname + '.png', dpi=200)
	plt.savefig('C:/Users/hagen/Dropbox/Universiteit van Amsterdam/J2S2 Thesis/visualisations/tsne/mpl_tsnescatterplot_' + plotname + '.png', dpi=200)
	plt.gcf().clear()

def getsimilars(word, month):
	df_similars = pd.DataFrame()

	# word1='must'
	# word2='have'

	model = getWord2VecModel(load='word2vec/models/w2v_model_all-' + month + '.model')
	similars = model.wv.most_similar(positive=[word], topn = 30)
	df_similars[month] = [words[0] for words in similars]
	df_similars['ratio-' + month] = [int((words[1] * 100)) for words in similars]
	return df_similars

def createTokensOfCsv():
	#list of list of comments, timeseparated
	li_allstrings = []

	folder = 'substring_mentions/mentions_trump/months/'
	for (dirpath, dirnames, filenames) in os.walk(folder):
		for filename in filenames:
			if filename.endswith('.csv'):
				df = pd.read_csv(folder + filename, encoding='utf-8')
				li_comments = []
				for index, comment in enumerate(df['comment']):
					#remove urls
					if 'http' in comment:
						comment = re.sub(r'https?:\/\/.*[\r\n]*', ' ', comment)
					if 'www.' in comment:
						comment = re.sub(r'www.*[\r\n]*', ' ', comment)
					#remove general threads because they clutter the dataset with copy-pastas
					if 'general' not in df['title'][index].lower():
						# print('General thread, discarded')
						# print(df['title'][index])
						li_comments.append(comment)
				li_comments = ' '.join(li_comments)
				print(filename)
				li_allstrings.append(li_comments)
	return li_allstrings
	#print(li_allstrings[:10])
	# for li_str_months in li_allstrings:
	# 	month_tokens = getTokens(li_strings=li_str_months, similaritytype='docs', stems=True)

#tokens = createTokensOfCsv()

# df_trumpthreads = pd.read_csv('substring_mentions/mentions_trump/trump_threads/trump_threads_15percent_30min.csv', encoding='utf-8')
# li_strings = df_trumpthreads['comment'].tolist()
# df_trumpthreads = ''

#both a stemmed and non-stemmed version
# words_stemmed = getTokens(li_strings, stemming=True)
# pickle.dump(words_stemmed, open('substring_mentions/mentions_trump/trump_threads/trump_threads_tokens_15percent_30min_stemmed.p', 'wb'))
# model = getWord2VecModel(train=words_stemmed, modelname='model_trump_threads_15percent_30min_stemmed')

# pickle.dump(words_stemmed, open('substring_mentions/mentions_trump/trump_threads/trump_threads_tokens_15percent_30min.p', 'wb'))

#words_stemmed = pickle.load(open('substring_mentions/mentions_trump/trump_threads/trump_threads_tokens_15percent_30min_stemmed.p', 'rb'))
# model = getWord2VecModel(load='word2vec/models/w2v_model_all-05-2017.model')
# print(model.most_similar(positive=['kekistan']))
# model = getWord2VecModel(load='word2vec/models/w2v_model_all-06-2017.model')
# print(model.most_similar(positive=['kekistan'], topn=30))
# model = getWord2VecModel(load='word2vec/models/w2v_model_all-07-2017.model')
# print(model.most_similar(positive=['kekistan']))
# model = getWord2VecModel(load='word2vec/models/w2v_model_all-08-2017.model')
# print(model.most_similar(positive=['kekistan']))

# df_trumpthreads = pd.read_csv('substring_mentions/mentions_trump/trump_threads/trump_threads_15percent_30min.csv', engine='python', encoding='utf-8')
# dates = df_trumpthreads['date_month'].unique()
# for month in dates:
# 	print('Creating DataFrame and stemming for ' + month)
# 	df_month = df_trumpthreads[df_trumpthreads['date_month'] == month]
# 	df_month.to_csv('substring_mentions/mentions_trump/trump_threads/trump_threads_15percent_30min_' + month + '.csv', encoding='utf-8')
# 	li_strings = df_month['comment'].tolist()
# 	words_stemmed = getTokens(li_strings, stemming=True)
# 	pickle.dump(words_stemmed, open('substring_mentions/mentions_trump/trump_threads/trump_threads_tokens_15percent_30min_stemmed_' + month + '.p', 'wb'))
# 	model = getWord2VecModel(train=li_strings, modelname='model_trump_threads_15percent_30min_stemmed' + month)

# folder = 'substring_mentions/mentions_trump/trump_threads/months/'
# for root, dirs, files in os.walk(folder):
# 	for filename in files:
# 		print(filename)
# 		words_stemmed = pickle.load(open(folder + filename, 'rb'))
# 		model = getWord2VecModel(train=words_stemmed, modelname='trumpthreads/model_' + filename)
# 		getTsneScatterPlot(model, plotname='model_trump_threads_15percent_30min_200minword_stemmed' + filename)