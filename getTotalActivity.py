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

def getTotalActivity(dateformat='months'):
	print('Connecting to database')
	conn = sqlite3.connect("../4plebs_pol_test_database.db")

	if dateformat == 'months':
		dateformat = '%Y-%m'
	elif dateformat == 'days':
		dateformat = '%Y-%m-%d'
	print('Beginning SQL query')
	dates = pd.read_sql_query("SELECT DISTINCT(timestamp) FROM poldatabase;", conn)
	print(dates)
	li_alldates = dates['timestamp'].values.tolist()
	li_alldates.sort()
	firstdate = li_alldates[0]
	lastdate = li_alldates[len(li_alldates) - 1]
	print(firstdate, lastdate)

	li_dates = []
	li_countposts = []

	headers=['date','posts']
	df_timethreads = pd.DataFrame(columns=headers)
	newtime = ''
	minquerydate = firstdate
	currenttime = datetime.fromtimestamp(minquerydate).strftime(dateformat)
	for timestamp in li_alldates:
		#print(timestamp)
		if timestamp != lastdate:
			newtime = datetime.fromtimestamp(timestamp).strftime(dateformat)
			#if there's a new date
			if currenttime != newtime:
				print('SQL query for ' + str(newtime))
				timestring = str(newtime)
				maxquerydate = timestamp
				df = pd.DataFrame
				df = pd.read_sql_query("SELECT COUNT(*)count FROM poldatabase WHERE timestamp > :firsttime AND timestamp < :lasttime;", conn, params={'dateheader': newtime, 'firsttime': minquerydate, 'lasttime': maxquerydate})
				
				li_dates.append(str(newtime))
				li_countposts.append(df['count'][0])
				
				minquerydate = timestamp
				currenttime = newtime

	df_timethreads['date'] = li_dates
	df_timethreads['posts'] = li_countposts
	print('Writing results to csv')
	df_timethreads.to_csv('all_activity.csv', index=False)
	print(df_timethreads)

getTotalActivity(dateformat='days')