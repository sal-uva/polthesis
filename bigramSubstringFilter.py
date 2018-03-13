
import sqlite3
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import nltk
import re
import operator
from datetime import datetime, timedelta
from collections import OrderedDict
from nltk.collocations import *
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

#HEADERS: num, subnum, thread_num, op, timestamp, timestamp_expired, preview_orig,
# preview_w, preview_h, media_filename, media_w, media_h, media_size,
# media_hash, media_orig, spoiler, deleted, capcode, email, name, trip,
# title, comment, sticky, locked, poster_hash, poster_country, exif

def colocationSubstringFilter(querystring, fullcomment = True, colocationamount = 1, windowsize = 4, outputlimit = 10, separateontime = False, timeseparator = 'days'):

	maxoutput = outputlimit

	print('Connecting to database')
	conn = sqlite3.connect("../4plebs_pol_test_database.db")

	print('Beginning SQL query for "' + querystring + '"')
	df = pd.read_sql_query("SELECT timestamp, comment FROM poldatabase WHERE lower(comment) LIKE ?;", conn, params=['%' + querystring + '%'])
	print('Writing results to csv')
	df.to_csv('mentions_' + querystring + '.csv')

	#take DataFrame columns and convert to list
	print('Converting DataFrame columns to lists')
	li_posts = df['comment'].tolist()
	li_timestamplist = df['timestamp'].tolist()
	#remove column header (first entry in list)
	li_posts = li_posts[1:]
	li_timestamplist = li_timestamplist[1:]

	print('Generating string for colocations')
	longstring = ''
	#dependent on 'timeseparater', create multiple string or just one string
	if separateontime == False:
		longstring = ''
		for comment in li_posts:
			longstring = longstring + str(comment) + ' '

	elif separateontime == True:
		di_timestrings = {}
		str_timeinterval = ''
		
		for index, timestamp in enumerate(li_timestamplist):
			if currenttimeinverval == datetime.fromtimestamp(li_timestamplist[index]).strftime("%Y-%m-%d"):
				str_timeinterval = str(str_timeinterval) + str(li_posts[index]) + ' '
			else:
				str_day = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
				di_timestrings[str_day] = str_timeinterval 		#put a threadnumber value as key with full string
				str_timeinterval = ''
				str_timeinterval = str(li_posts[index])
				currenttimeinverval = str_day
	
	bigram_measures = nltk.collocations.BigramAssocMeasures()

	#all text to lowercase
	longstring = longstring.lower()

	regex = re.compile("[^a-zA-Z]")		#excludes numbers, might have to revise this
	longstring = regex.sub(" ",longstring)

	tokenizer = RegexpTokenizer(r"\w+")
	tmptokens = tokenizer.tokenize(longstring)

	tokens = []
	forbiddenwords = ['www','youtube','com','watch', 'http', 'https', 'v', 'en', 'wikipedia', 'wiki', 'org']

	print('Creating filtered tokens (i.e. excluding stopwords and forbiddenwords)')
	for word in tmptokens:
		if word not in stopwords.words("english"):
			if word not in forbiddenwords:
				match = re.search(word, r'(\d{9})')
				if not match:		#if it's a post number
					tokens.append(word)

	print('Generating colocations')
	if separateontime == False:
		if colocationamount == 1:
			finder = BigramCollocationFinder.from_words(tokens, window_size=windowsize)
		if colocationamount == 2:
			finder = TrigramCollocationFinder.from_words(tokens, window_size=windowsize)

	elif separateontime == True:
		for timeinterval in li_timeintervals:
			if colocationamount == 1:
				finder = BigramCollocationFinder.from_words(tokens[timeinterval], window_size=windowsize)
			if colocationamount == 2:
				finder = TrigramCollocationFinder.from_words(tokens[timeinterval], window_size=windowsize)

	colocations = sorted(finder.ngram_fd.items(), key=operator.itemgetter(1), reverse=True)[0:outputlimit]
	print(colocations)
	return(colocations)

colocationSubstringFilter('netherlands', fullcomment=True, colocationamount=1, windowsize = 4, outputlimit = 20, separateontime = False, timeseparator = 'days')

print('finished')