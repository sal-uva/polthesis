from __future__ import print_function
import sqlite3
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import re
import os
import similarities
import createHistogram
import nltk
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

#HEADERS: num, subnum, thread_num, op, timestamp, timestamp_expired, preview_orig,
# preview_w, preview_h, media_filename, media_w, media_h, media_size,
# media_hash, media_orig, spoiler, deleted, capcode, email, name, trip,
# title, comment, sticky, locked, poster_hash, poster_country, exif

# full db: 4plebs_pol_18_03_2018
# full table: poldatabase_18_03_2018
# test db: 4plebs_pol_test_database
# test table: poldatabase

def substringFilter(inputstring, histogram = False, stringintitle = False, inputtime = 'months', normalised=False, writetext=False, docsimilarity = False, wordclusters = False, similaritytype=None):
	querystring = inputstring.lower()

	print('Connecting to database')
	conn = sqlite3.connect("../4plebs_pol_18_03_2018.db")

	print('Beginning SQL query for "' + querystring + '"')
	# if stringintitle == False:
	# 	df = pd.read_sql_query("SELECT timestamp, comment FROM poldatabase_18_03_2018 WHERE lower(comment) LIKE ?;", conn, params=['%' + querystring + '%'])
	# else:
	# 	df = pd.read_sql_query("SELECT timestamp, title FROM poldatabase_18_03_2018 WHERE lower(title) LIKE ?;", conn, params=['%' + querystring + '%'])

	# print('Writing results to csv')
	# if '/' in querystring:
	# 	querystring = re.sub(r'/', '', querystring)
	# else:
	# 	querystring = querystring
	# df.to_csv('substring_mentions/mentions_' + querystring + '.csv')

	#FOR DEBUGGING PURPOSES:
	df = pd.read_csv('substring_mentions/mentions_trump.csv')

	if writetext == True:
		df = df.sort_values(by=['timestamp'])
		df_parsed = pd.DataFrame(columns=['comments','time'])
		df_parsed['comments'] = df['comment']
		#note: make day seperable later
		if inputtime == 'months':
			df_parsed['time'] = [datetime.strftime(datetime.fromtimestamp(i), "%m-%Y") for i in df['timestamp']]
		elif inputtime == 'weeks':
			df_parsed['time'] = [datetime.strftime(datetime.fromtimestamp(i), "%W-%Y") for i in df['timestamp']]
		elif inputtime == 'days':
			df_parsed['time'] = [datetime.strftime(datetime.fromtimestamp(i), "%d-%m-%Y") for i in df['timestamp']]
		
		df_parsed['comments'] = [re.sub(r'>', ' ', z) for z in df_parsed['comments']]
		df_parsed = df_parsed.sort_values(by=['time'])
		#print(df_parsed['comments'])

		#write text file for separate months
		currenttime = df_parsed['time'][1]
		oldindex = 1

		li_str_timeseparated = []
		li_str_full = []
		li_stringdates = []
		#create text files for each month
		for index, distincttime in enumerate(df_parsed['time']):
			#if the timestring is different from before, or the end of the column is reached
			if distincttime != currenttime or index == (len(df_parsed['time']) - 1):
				print(currenttime, distincttime)
				
				df_sliced = df_parsed[oldindex:index]
				print(df_sliced)
				df_sliced.to_csv('substring_mentions/' + querystring + '_' + currenttime + '.csv', encoding='utf-8')
				string, li_strings = writeToText(df_sliced, querystring, currenttime)
				li_str_timeseparated.append(string)
				li_str_full.append(li_strings)
				li_stringdates.append(currenttime)
				oldindex = index + 1
				currenttime = distincttime				

	if similaritytype == 'docs' or similaritytype == 'words':
		if similaritytype == 'docs':
			words_stemmed = similarities.getTokens(li_str_timeseparated, li_stringdates, similaritytype)
			similarities.getDocSimilarity(li_str_timeseparated, words_stemmed, li_stringdates, querystring)
		elif similaritytype == 'words':
			words_stemmed = similarities.getTokens(li_str_full, li_stringdates, similaritytype)
			similarities.getWordSimilarity(words_stemmed)

	if histogram == True:
		createHistogram(df, querystring, inputtime, normalised)

def writeToText(inputdf, querystring, currenttime):
	directory = 'substring_mentions/longstring_' + querystring
	if not os.path.exists(directory):
		os.makedirs(directory)
	txtfile = open('substring_mentions/longstring_' + querystring + '/longstring_' + querystring + '_' + currenttime + '.txt', 'w', encoding='utf-8')
	str_keyword = ''
	li_str = []
	for item in inputdf['comments']:
		item = item.lower()
		regex = re.compile("[^a-zA-Z \.\n]")		#excludes numbers, might have to revise this
		item = regex.sub("", item)
		txtfile.write("%s" % item)
		str_keyword = str_keyword + item
		li_str.append(item)
	return str_keyword, li_str

li_querywords = ['trump']

for word in li_querywords:
	result = substringFilter(word, histogram = False, stringintitle = False, inputtime='weeks', normalised=True, writetext=True, similaritytype=None)	#returns tuple with df and input string
print('finished')