
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

def getNgrams(querystring, fullcomment = True, colocationamount = 1, windowsize = 4, outputlimit = 10, separateontime = False, timeseparator = 'days'):

	maxoutput = outputlimit

	print('Connecting to database')
	conn = sqlite3.connect("../4plebs_pol_test_database.db")

	print('Beginning SQL query for "' + querystring + '"')
	df = pd.read_sql_query("SELECT timestamp, comment FROM poldatabase WHERE lower(comment) LIKE ?;", conn, params=['%' + querystring + '%'])
	print('Writing results to csv')
	df.to_csv('substring_mentions/mentions_' + querystring + '.csv')

	#take DataFrame columns and convert to list
	print('Converting DataFrame columns to lists')
	li_posts = df['comment'].tolist()
	li_timestamplist = df['timestamp'].tolist()
	#remove column header (first entry in list)
	li_posts = li_posts[1:]
	li_timestamplist = li_timestamplist[1:]

	print('Generating string for colocations')
	forbiddenwords = ['www','youtube','com','watch', 'http', 'https', 'v', 'en', 'wikipedia', 'wiki', 'org']

	#dependent on 'timeseparator', create multiple string or just one string
	if separateontime == False:
		longstring = ''
		longstring = ' '.join(li_posts)
		
		bigram_measures = nltk.collocations.BigramAssocMeasures()

		#all text to lowercase
		longstring = longstring.lower()

		regex = re.compile("[^a-zA-Z]")		#excludes numbers, might have to revise this
		longstring = regex.sub(" ",longstring)

		tokenizer = RegexpTokenizer(r"\w+")
		tmptokens = tokenizer.tokenize(longstring)

		tokens = []

		print('Creating filtered tokens (i.e. excluding stopwords and forbiddenwords)')
		for word in tmptokens:
			if word not in stopwords.words("english"):
				if word not in forbiddenwords:
					match = re.search(word, r'(\d{9})')
					if not match:		#if it's a post number
						tokens.append(word)

		print('Generating colocations')
		if colocationamount == 1:
			finder = BigramCollocationFinder.from_words(tokens, window_size=windowsize)
		if colocationamount == 2:
			finder = TrigramCollocationFinder.from_words(tokens, window_size=windowsize)

		fullcolocations = sorted(finder.ngram_fd.items(), key=operator.itemgetter(1), reverse=True)[0:outputlimit]

	elif separateontime == True:
		di_timestrings = {}
		str_timedivided = ''
		str_timeinterval = ''
		currenttimeinverval = ''
		
		for index, timestamp in enumerate(li_timestamplist):
			#check if the post is the same date as the previous post
			if currenttimeinverval == datetime.fromtimestamp(li_timestamplist[index]).strftime("%Y-%m-%d"):
				str_timedivided = str(str_timedivided) + str(li_posts[index]) + ' '
			#if its a new day/month, make a new dict entry
			else:
				di_timestrings[str_timeinterval] = str_timedivided
				str_timeinterval = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
				str_timedivided = str(li_posts[index])
				currenttimeinverval = str_timeinterval
		#print(di_timestrings)
		
		di_time_ngrams = {}
		for key, value in di_timestrings.items():
			#all text to lowercase
			longstring = value.lower()

			regex = re.compile("[^a-zA-Z]")		#excludes numbers, might have to revise this
			longstring = regex.sub(" ",longstring)

			tokenizer = RegexpTokenizer(r"\w+")
			tmptokens = tokenizer.tokenize(longstring)

			tokens = []

			print('Creating filtered tokens (i.e. excluding stopwords and forbiddenwords)')
			for word in tmptokens:
				if word not in stopwords.words("english"):
					if word not in forbiddenwords:
						match = re.search(word, r'(\d{9})')
						if not match:		#if it's a post number
							tokens.append(word)
			if colocationamount == 1:
				finder = BigramCollocationFinder.from_words(tokens, window_size=windowsize)
			if colocationamount == 2:
				finder = TrigramCollocationFinder.from_words(tokens, window_size=windowsize)
			colocations = sorted(finder.ngram_fd.items(), key=operator.itemgetter(1), reverse=True)[0:outputlimit]
			di_time_ngrams[key] = colocations
			print(colocations)

		fullcolocations = str(di_time_ngrams)

	print('Writing restults to textfile')
	write_handle = open('colocations/' + str(querystring) + '-colocations.txt',"w")
	write_handle.write(str(fullcolocations))
	write_handle.close()
	
	print(fullcolocations)
	return(fullcolocations)

getNgrams('nigger', fullcomment=True, colocationamount=2, windowsize=4, outputlimit=3, separateontime=True)

print('finished')