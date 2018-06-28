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

# test db: 4plebs_pol_test_database
# test table: poldatabase
# full db: 4plebs_pol_18_03_2018
# full table: poldatabase_18_03_2018

def getTotalActivity(dateformat='months', posts=False, threads=False):
	print('Connecting to database')
	conn = sqlite3.connect("../4plebs_pol_18_03_2018.db")

	if posts:
		df = pd.read_sql_query(""" 	SELECT date_month, count(*)count FROM pol_content
									GROUP BY date_month
									""", conn)
		print(df)
		df.to_csv('metadata/comments_per_month.csv')

	if threads:
		df = pd.read_sql_query("""	SELECT date_month, count(*)count FROM
										(SELECT DISTINCT(thread_num), date_month FROM pol_content)
									GROUP BY date_month
									""", conn)
		print(df)
		df.to_csv('metadata/threads_per_month.csv')


	if 1==2:

		if dateformat == 'months':
			dateformat = '%Y-%m'
		elif dateformat == 'days':
			dateformat = '%Y-%m-%d'

		print('Beginning SQL query')

		li_dates = []

		# dates = pd.read_sql_query("SELECT MIN(timestamp)mintimestamp, MAX(timestamp)maxtimestamp FROM poldatabase_18_03_2018;", conn)
		# print(dates)
		firstdate = 1378739071
		lastdate = 1521370386
		# firstdate = dates['mintimestamp'][0]
		# lastdate = dates['maxtimestamp'][0]
		print(firstdate, lastdate)
		
		li_dates = []
		li_countposts = []

		headers=['date','posts']
		df_timethreads = pd.DataFrame(columns=headers)
		newtime = ''
		minquerydate = firstdate
		currenttime = datetime.fromtimestamp(minquerydate).strftime(dateformat)
		timestamp = firstdate
		while timestamp < lastdate:
			timestamp = timestamp + 1
			#print(timestamp)
			if timestamp != lastdate:
				newtime = datetime.fromtimestamp(timestamp).strftime(dateformat)
				#if there's a new date
				if currenttime != newtime:
					print('SQL query for ' + str(newtime))
					timestring = str(newtime)
					maxquerydate = timestamp
					print(minquerydate, maxquerydate)
					
					df = pd.read_sql_query("SELECT COUNT(*)count FROM poldatabase_18_03_2018 WHERE (timestamp BETWEEN ? AND ?);", conn, params=[minquerydate, maxquerydate])
					print(df)
					print(df['count'])
					li_dates.append(str(newtime))
					li_countposts.append(df['count'][0])
					print(li_countposts)
					minquerydate = timestamp
					currenttime = newtime

		df_timethreads['date'] = li_dates
		df_timethreads['posts'] = li_countposts
		print('Writing results to csv')
		df_timethreads.to_csv('all_activity.csv', index=False)
		print(df_timethreads)

#getTotalActivity(threads=True)

def addDatesToDb():
	conn = sqlite3.connect("../4plebs_pol_18_03_2018.db")
	cursor=conn.cursor()
	if 1 == 2:
		cursor.execute("""	CREATE TABLE 'pol_content' AS
							SELECT num, thread_num, timestamp,
							strftime('%Y-%m-%d %H:%M:%S', datetime(timestamp, 'unixepoch')) as date_full,
							op, title, comment, timestamp_expired, media_filename, media_size, media_hash, media_orig, spoiler, deleted, capcode, email, name, trip, sticky, locked, poster_hash, poster_country, exif,
							strftime('%Y-%m', datetime(timestamp, 'unixepoch')) as date_month,
							strftime('%Y-%m-%d', datetime(timestamp, 'unixepoch')) as date_day,
							strftime('%Y-%W', datetime(timestamp, 'unixepoch')) as date_week
							FROM 'poldatabase_18_03_2018';""")

def calculateAverageAnon():
	li_avposts = [0.5]
	for i in [n for n in range(1, 16)]:
		li_avposts.append(i)
	#li_avposts.append([i for i in [n for n in range(1, 15)]])
	print(li_avposts)
	av_anons = [int(3468140 / (i * 28)) for i in li_avposts]
	x = li_avposts
	print(av_anons, x)
	fig = plt.figure(figsize=(11, 8))
	fig.set_dpi(100)
	ax = fig.add_subplot(111)
	ax.plot(x, av_anons)
	ax.set_ylim(bottom=0)
	#plt.xlim(-0.5,len(x)-.5)
	ax.set_xticks(x)
	ax.grid(color='#e5e5e5',linestyle='dashed', linewidth=.6)
	ax.set_ylabel('Amount of committed anons needed')
	ax.set_xlabel('Average posts per anon per day')
	ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
	ax.set_xticklabels(labels = [str(i) for i in li_avposts])
	plt.title('Anons/posts needed for the 3,468,140 posts in February 2018')
	# for label in ax.xaxis.get_ticklabels()[::2]:
	# 	label.set_visible(False)
	plt.savefig('../visualisations/anon_estimation_feb2018.png', dpi='figure',bbox_inches='tight')
	plt.savefig('../visualisations/anon_estimation_feb2018.svg', dpi='figure',bbox_inches='tight')
	# plt.show()
