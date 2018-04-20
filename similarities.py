from __future__ import print_function
import sqlite3
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import re
import os
import nltk
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
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
from matplotlib import pyplot

def getTokens(li_strings='', dates=None, similaritytype='docs', stems=False):
	print('imported')
	#do some cleanup: only alphabetic characters, no stopwords
	# create separate stemmed tokens, to which the full strings will be compared to:
	li_comments_stemmed = []
	len_comments = len(li_strings)
	print(len(li_strings))
	print('Creating list of tokens per monthly document')
	for index, comment in enumerate(li_strings):
		#print(comment)
		# if docs, work with a list, if words, work with a list of list
		if similaritytype == 'words':
			if isinstance(comment, str):
				li_comment_stemmed = []
				li_comment_stemmed = tokeniserAndStemmer(comment)
				li_comments_stemmed.append(li_comment_stemmed)
				#print(li_stemmed)
		elif similaritytype == 'docs':
			words_stemmed = tokeniserAndStemmer(string, steming=stems)
			li_stemmed.extend(words_stemmed)
		if index % 1000 == 0:
			print('Stemming/tokenising finished for string ' + str(index) + '/' + str(len_comments))
	print(len(li_comments_stemmed))
	return li_comments_stemmed

def getDocSimilarity(li_strings, words_stemmed, dates, querystring):
	# look up iterative k-means
	print('Creating tf-idf vector of input documents')
	#prepare vectorizer
	tfidf_vectorizer = TfidfVectorizer(max_df = 0.9, min_df = 0.1, stop_words='english', analyzer='word', use_idf=True, tokenizer=tokeniserAndStemmer)
	#create tf_idf vectors of month-separated comments
	tfidf_matrix = tfidf_vectorizer.fit_transform(li_strings)
	print(tfidf_matrix)

	# cosine similarity used before:
	# similarityvector = (tfidf_matrix * tfidf_matrix.T).A
	# print(similarityvector)
	# print(type(similarityvector))

	# print('Writing similarity vector to csv')
	# df_similarity = pd.DataFrame(similarityvector, index=dates, columns=dates)
	# df_similarity.to_csv('substring_mentions/tfidf_' + querystring + '.csv')

	print('Calculating document similarities')
	terms = tfidf_vectorizer.get_feature_names()
	print(terms)
	dist = 1 - cosine_similarity(tfidf_matrix)
	print(dist)

	num_clusters = 7

	# create new K-means clusters
	k_means = KMeans(n_clusters = num_clusters)
	k_means.fit(tfidf_matrix)
	clusters = k_means.labels_.tolist()
	print(clusters)
	joblib.dump(k_means,  'clusters/doc_cluster_' + querystring + '.pkl')

	# loading existing clusters for debugging/testing
	# k_means = joblib.load('doc_cluster.pkl')
	# clusters = k_means.labels_.tolist()
	di_clusters = {'dates': dates, 'text': li_strings, 'cluster': clusters}

	df_kclusters = pd.DataFrame(di_clusters, index=[clusters], columns = ['dates', 'cluster'])
	df_kclusters.to_csv('clusters/cluster_'+ querystring + '.csv')

	#sort cluster centers by proximity to centroid
	order_centroids = k_means.cluster_centers_.argsort()[:, ::-1] 

	di_cluster_names = {}
	for i in range(num_clusters):
		clusterstring = ''
		print("Cluster %d:" % i),
		for ind in order_centroids[i, :7]:
			print(' %s' % terms[ind])
			clusterstring += ' ' + terms[ind]
		di_cluster_names[i] = clusterstring
	
	di_cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e', 5: '#0012e0', 6: '#e0d100'}
	MDS()

	# convert two components as we're plotting points in a two-dimensional plane
	# "precomputed" because we provide a distance matrix
	# we will also specify `random_state` so the plot is reproducible.
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

def getWord2VecModel(train='', load='', modelname=''):
	if train != '':
		# train model
		# neighbourhood?
		model = Word2Vec(train, min_count=2)
		# pickle the entire model to disk, so we can load&resume training later
		model.save('word2vec/w2v_model_' + modelname + '.model')
		#store the learned weights, in a format the original C tool understands
		model.wv.save_word2vec_format('word2vec/w2v_model_' + modelname + '.model.bin', binary=True)
		return model
	elif load != '':
		# or, import word weights created by the (faster) C word2vec
		# this way, you can switch between the C/Python toolkits easily
		model = Word2Vec.load(load)
		return model
	
def showPCAGraph(model):
	# use t-sne!
	# PCA is more effective for 'importance' of words

	# fit a 2d PCA model to the vectors
	X = model[model.wv.vocab]
	pca = PCA(n_components=10)
	result = pca.fit_transform(X)
	# create a scatter plot of the projection

	pyplot.scatter(result[:, 0], result[:, 1])
	words = list(model.wv.vocab)
	for i, word in enumerate(words):
		pyplot.annotate(word, xy=(result[i, 0], result[i, 1]), size=6)
	plt.rcParams.update({'font.size': 3})
	pyplot.show()


def tokeniserAndStemmer(string, stemming=False):
	stemmer = SnowballStemmer("english")
	tokens = [word for sent in nltk.sent_tokenize(string) for word in nltk.word_tokenize(sent)]
	li_filtered_tokens = []
	# filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
	for token in tokens:
		if re.search('[a-zA-Z]', token):
			if len(token) > 2 and len(token) < 20:
				if token not in stopwords.words('english'):
					token = token.lower()
					li_filtered_tokens.append(token)
	if stemming == True:
		stems = [stemmer.stem(t) for t in li_filtered_tokens]
		return stems
	else:
		return li_filtered_tokens

# some calls for these function come from substring
def getSimilaritiesFromCsv(csvdoc='', modelname = ''):
	df = pd.read_csv(csvdoc, encoding='utf-8')
	li_strings = []
	for comment in df['comment']:
		li_strings.append(comment)
	words_stemmed = getTokens(li_strings, similaritytype='words', stems=False)
	#print(words_stemmed[:100])
	#df_stemmedwords = pd.DataFrame(words_stemmed)
	#USE PICKLE
	df_stemmedwords.to_csv('test_stemmed.csv', encoding='utf-8')

	model = getWord2VecModel(train=words_stemmed, modelname=modelname)
	# model = getWord2VecModel(load=modelname)
	#showPCAGraph(model)
	# similars = model.most_similar(positive=['btfo'], topn = 20)
	# print(similars)
	# similars = model.similar_by_vector(model['hillari'] + model['polit'])
	# print(similars)

# getSimilaritiesFromCsv(modelname='word2vec/w2v_model_all-05-2015.model')

#getSimilaritiesFromCsv('substring_mentions/all_01-2016.csv', modelname='all-01-2016')