from __future__ import print_function
import sqlite3
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
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

def substringFilter(inputstring = 'all', histogram = False, mintime = 0, maxtime = 0, stringintitle = False, inputtime = 'months', normalised = False, writetext = False, docsimilarity = False, wordclusters = False, similaritytype = None):
	querystring = inputstring.lower()

	print('Connecting to database')
	conn = sqlite3.connect("../4plebs_pol_18_03_2018.db")

	print('Beginning SQL query for "' + querystring + '"')

	#if you get all comments, filter on 
	if querystring == 'all':
		querystring = querystring + '-' + str(datetime.strftime(datetime.fromtimestamp(mintime), "%m-%Y"))
		df = pd.read_sql_query("SELECT timestamp, comment FROM poldatabase_18_03_2018 WHERE timestamp > ? AND timestamp < ?;", conn, params=[mintime, maxtime])
	#look for string in subject
	elif stringintitle == False:
		df = pd.read_sql_query("SELECT timestamp, comment FROM poldatabase_18_03_2018 WHERE lower(comment) LIKE ?;", conn, params=['%' + querystring + '%'])
	#look for sting in comment body (default)
	else:
		df = pd.read_sql_query("SELECT timestamp, title, comment FROM poldatabase_18_03_2018 WHERE lower(title) LIKE ?;", conn, params=['%' + querystring + '%'])

	print('Writing results to csv')
	if '/' in querystring:
		querystring = re.sub(r'/', '', querystring)
	else:
		querystring = querystring
	df.to_csv('substring_mentions/mentions_' + querystring + '.csv')

	#FOR DEBUGGING PURPOSES:
	#df = pd.read_csv('substring_mentions/mentions_all.csv')

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
		
		#df_parsed['comments'] = [re.sub(r'>', ' ', z) for z in df_parsed['comments']]
		df_parsed = df_parsed.sort_values(by=['time'])
		#print(df_parsed['comments'])

		#write text file for separate months
		currenttime = df_parsed['time'][1]
		oldindex = 1

		li_str_timeseparated = []
		li_str_full = []
		li_stringdates = []
		#create text files for each month for WordTree maps
		for index, distincttime in enumerate(df_parsed['time']):
			#if the timestring is different from before, or the end of the column is reached
			if distincttime != currenttime or index == (len(df_parsed['time']) - 1):
				print(currenttime, distincttime)
				
				df_sliced = df_parsed[oldindex:index]
				#print(df_sliced)
				df_sliced.to_csv('substring_mentions/' + querystring + '_' + currenttime + '.csv', encoding='utf-8')
				string, li_strings = writeToText(df_sliced, querystring, currenttime)
				li_str_timeseparated.append(string)
				li_str_full.append(li_strings)
				li_stringdates.append(currenttime)
				oldindex = index + 1
				currenttime = distincttime				

	if similaritytype != None:
		if similaritytype == 'docs' or similaritytype == 'words':
			if similaritytype == 'docs':
				words_stemmed = similarities.getTokens(li_str_timeseparated, li_stringdates, similaritytype)
				similarities.getDocSimilarity(li_str_timeseparated, words_stemmed, li_stringdates, querystring)
			elif similaritytype == 'words':
				words_stemmed = similarities.getTokens(li_str_full, li_stringdates, similaritytype)
				similarities.getWordSimilarity(words_stemmed)

	if histogram == True:
		createHistogram.createHistogram(df, querystring, inputtime, normalised)
	df_return = pd.DataFrame()
	df_return['comment'] = df['comment']
	#print(df)
	return df

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

#i_querywords = ['all']

#may 2015: (1430438400,1433116799),
#timestamps for Jun 2015 - Apr 2018
#li_times = [(1433116800,1435708799),(1435708800,1438387199),(1438387200,1441065599),(1441065600,1443657599),(1443657600,1446335999),(1446336000,1448927999),(1448928000,1451606399),(1451606400,1454284799),(1454284800,1456790399),(1456790400,1459468799),(1459468800,1462060799),(1462060800,1464739199),(1464739200,1467331199),(1467331200,1470009599),(1470009600,1472687999),(1472688000,1475279999),(1475280000,1477958399),(1477958400,1480550399),(1480550400,1483228799),(1483228800,1485907199),(1485907200,1488326399),(1488326400,1491004799),(1491004800,1493596799),(1493596800,1496275199),(1496275200,1498867199),(1498867200,1501545599),(1501545600,1504223999),(1504224000,1506815999),(1506816000,1509494399),(1509494400,1512086399),(1512086400,1514764799),(1514764800,1517443199),(1517443200,1519862399),(1519862400,1522540799),(1522540800,1525132799)]

#used for model, oct 2015 t/m feb 2016:
#li_times = [(1443657600,1446335999),(1446336000,1448927999),(1448928000,1451606399),(1451606400,1454284799),(1454284800,1456790399)]

#done:
#(1433116800,1435708799),(1435708800,1438387199),(1438387200,1441065599),(1441065600,1443657599),(1443657600,1446335999),(1446336000,1448927999),(1448928000,1451606399),(1451606400,1454284799),(1454284800,1456790399),(1456790400,1459468799),(1459468800,1462060799),(1462060800,1464739199),(1464739200,1467331199),(1467331200,1470009599),(1470009600,1472687999),(1472688000,1475279999)

# individual monthly models:
li_times = [(1475280000,1477958399),(1477958400,1480550399),(1480550400,1483228799),(1483228800,1485907199),(1485907200,1488326399),(1488326400,1491004799),(1491004800,1493596799),(1493596800,1496275199),(1496275200,1498867199),(1498867200,1501545599),(1501545600,1504223999),(1504224000,1506815999),(1506816000,1509494399),(1509494400,1512086399),(1512086400,1514764799),(1514764800,1517443199),(1517443200,1519862399),(1519862400,1522540799),(1522540800,1525132799)]

def doCode():
	for tpl_time in li_times:
		result = substringFilter(inputstring='all', mintime = tpl_time[0], maxtime = tpl_time[1], histogram = False, stringintitle = False, inputtime='months', normalised=True, writetext=False)	#returns tuple with df and input string
		print('starting similarities')
		datestring = datetime.strftime(datetime.fromtimestamp(tpl_time[0]), "%m-%Y")
		similarities.getSimilaritiesFromCsv(result, modelname='all-' + datestring)

# substringFilter(inputstring='drumpf', histogram = True, stringintitle = False, inputtime='months', normalised=True, writetext=False)
# substringFilter(inputstring='god emperor', histogram = True, stringintitle = False, inputtime='months', normalised=True, writetext=False)
# substringFilter(inputstring='we must', histogram = True, stringintitle = False, inputtime='months', normalised=True, writetext=False)
doCode()