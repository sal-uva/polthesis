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

#HEADERS
# pol_content
# num, subnum, thread_num, op, timestamp, timestamp_expired, preview_orig,
# preview_w, preview_h, media_filename, media_w, media_h, media_size,
# media_hash, media_orig, spoiler, deleted, capcode, email, name, trip,
# title, comment, sticky, locked, poster_hash, poster_country, exif

# trump_threads
# threadno thread_count  trump_count  trump_density

# full db: 4plebs_pol_18_03_2018
# full table: poldatabase_18_03_2018
# test db: 4plebs_pol_test_database
# test table: poldatabase

#not very efficient, but having these lists helps since I'm constantly reusing it and keeps me away from datetime library...
li_times_months = [(1433116800,1435708799),(1435708800,1438387199),(1438387200,1441065599),(1441065600,1443657599),(1443657600,1446335999),(1446336000,1448927999),(1448928000,1451606399),(1451606400,1454284799),(1454284800,1456790399),(1456790400,1459468799),(1459468800,1462060799),(1462060800,1464739199),(1464739200,1467331199),(1467331200,1470009599),(1470009600,1472687999),(1472688000,1475279999),(1475280000,1477958399),(1477958400,1480550399),(1480550400,1483228799),(1483228800,1485907199),(1485907200,1488326399),(1488326400,1491004799),(1491004800,1493596799),(1493596800,1496275199),(1496275200,1498867199),(1498867200,1501545599),(1501545600,1504223999),(1504224000,1506815999),(1506816000,1509494399),(1509494400,1512086399),(1512086400,1514764799),(1514764800,1517443199),(1517443200,1519862399),(1519862400,1522540799)]
li_labels_months = ['06-15', '07-15', '08-15', '09-15', '10-15', '11-15', '12-15', '01-16', '02-16', '03-16', '04-16', '05-16', '06-16', '07-16', '08-16', '09-16', '10-16', '11-16', '12-16', '01-17', '02-17', '03-17', '04-17', '05-17', '06-17', '07-17', '08-17', '09-17', '10-17', '11-17', '12-17','01-18', '02-18', '03-18']
li_times_weeks = [(1417392000, 1417996799), (1417996800, 1418601599), (1418601600, 1419206399), (1419206400, 1419811199), (1419811200, 1420415999), (1420416000, 1421020799), (1421020800, 1421625599), (1421625600, 1422230399), (1422230400, 1422835199), (1422835200, 1423439999), (1423440000, 1424044799), (1424044800, 1424649599), (1424649600, 1425254399), (1425254400, 1425859199), (1425859200, 1426463999), (1426464000, 1427068799), (1427068800, 1427673599), (1427673600, 1428278399), (1428278400, 1428883199), (1428883200, 1429487999), (1429488000, 1430092799), (1430092800, 1430697599), (1430697600, 1431302399), (1431302400, 1431907199), (1431907200, 1432511999), (1432512000, 1433116799), (1433116800, 1433721599), (1433721600, 1434326399), (1434326400, 1434931199), (1434931200, 1435535999), (1435536000, 1436140799), (1436140800, 1436745599), (1436745600, 1437350399), (1437350400, 1437955199), (1437955200, 1438559999), (1438560000, 1439164799), (1439164800, 1439769599), (1439769600, 1440374399), (1440374400, 1440979199), (1440979200, 1441583999), (1441584000, 1442188799), (1442188800, 1442793599), (1442793600, 1443398399), (1443398400, 1444003199), (1444003200, 1444607999), (1444608000, 1445212799), (1445212800, 1445817599), (1445817600, 1446422399), (1446422400, 1447027199), (1447027200, 1447631999), (1447632000, 1448236799), (1448236800, 1448841599), (1448841600, 1449446399), (1449446400, 1450051199), (1450051200, 1450655999), (1450656000, 1451260799), (1451260800, 1451865599), (1451865600, 1452470399), (1452470400, 1453075199), (1453075200, 1453679999), (1453680000, 1454284799), (1454284800, 1454889599), (1454889600, 1455494399), (1455494400, 1456099199), (1456099200, 1456703999), (1456704000, 1457308799), (1457308800, 1457913599), (1457913600, 1458518399), (1458518400, 1459123199), (1459123200, 1459727999), (1459728000, 1460332799), (1460332800, 1460937599), (1460937600, 1461542399), (1461542400, 1462147199), (1462147200, 1462751999), (1462752000, 1463356799), (1463356800, 1463961599), (1463961600, 1464566399), (1464566400, 1465171199), (1465171200, 1465775999), (1465776000, 1466380799), (1466380800, 1466985599), (1466985600, 1467590399), (1467590400, 1468195199), (1468195200, 1468799999), (1468800000, 1469404799), (1469404800, 1470009599), (1470009600, 1470614399), (1470614400, 1471219199), (1471219200, 1471823999), (1471824000, 1472428799), (1472428800, 1473033599), (1473033600, 1473638399), (1473638400, 1474243199), (1474243200, 1474847999), (1474848000, 1475452799), (1475452800, 1476057599), (1476057600, 1476662399), (1476662400, 1477267199), (1477267200, 1477871999), (1477872000, 1478476799), (1478476800, 1479081599), (1479081600, 1479686399), (1479686400, 1480291199), (1480291200, 1480895999), (1480896000, 1481500799), (1481500800, 1482105599), (1482105600, 1482710399), (1482710400, 1483315199), (1483315200, 1483919999), (1483920000, 1484524799), (1484524800, 1485129599), (1485129600, 1485734399), (1485734400, 1486339199), (1486339200, 1486943999), (1486944000, 1487548799), (1487548800, 1488153599), (1488153600, 1488758399), (1488758400, 1489363199), (1489363200, 1489967999), (1489968000, 1490572799), (1490572800, 1491177599), (1491177600, 1491782399), (1491782400, 1492387199), (1492387200, 1492991999), (1492992000, 1493596799), (1493596800, 1494201599), (1494201600, 1494806399), (1494806400, 1495411199), (1495411200, 1496015999), (1496016000, 1496620799), (1496620800, 1497225599), (1497225600, 1497830399), (1497830400, 1498435199), (1498435200, 1499039999), (1499040000, 1499644799), (1499644800, 1500249599), (1500249600, 1500854399), (1500854400, 1501459199), (1501459200, 1502063999), (1502064000, 1502668799), (1502668800, 1503273599), (1503273600, 1503878399), (1503878400, 1504483199), (1504483200, 1505087999), (1505088000, 1505692799), (1505692800, 1506297599), (1506297600, 1506902399), (1506902400, 1507507199), (1507507200, 1508111999), (1508112000, 1508716799), (1508716800, 1509321599), (1509321600, 1509926399), (1509926400, 1510531199), (1510531200, 1511135999), (1511136000, 1511740799), (1511740800, 1512345599), (1512345600, 1512950399), (1512950400, 1513555199), (1513555200, 1514159999), (1514160000, 1514764799), (1514764800, 1515369599), (1515369600, 1515974399), (1515974400, 1516579199), (1516579200, 1517183999), (1517184000, 1517788799), (1517788800, 1518393599), (1518393600, 1518998399), (1518998400, 1519603199), (1519603200, 1520207999), (1520208000, 1520812799), (1520812800, 1521417599)]
li_labels_weeks = []

def substringFilter(querystring='all', querystring2 = '', histogram = False, mintime = 0, maxtime = 0, stringintitle = False, inputtime = 'months', inmonth='', intrumpthreads = False, normalised = True, writetext = False, docsimilarity = False, wordclusters = False, similaritytype = None, debug=False):

	"""  
	Keywords:
	inmonth: only get posts that are in a certain month. Notes as string (e.g. 2014-01)

	"""

	querystring = querystring.lower()

	print('Connecting to database')

	#connect to a smaller databse for debug purposes
	if debug:
		conn = sqlite3.connect("../4plebs_pol_test_database.db")
	else:
		conn = sqlite3.connect("../4plebs_pol_18_03_2018.db")

	#if the trump-threads csv isn't used
	if intrumpthreads == False:
		#if you get all comments, filter on 
		if querystring == 'all':
			querystring = querystring + '-' + inmonth
			if inmonth != '':
				print('Beginning SQL query for all posts in ' + inmonth)
				df = pd.read_sql_query("SELECT timestamp, comment, title, num, date_full, date_month FROM pol_content WHERE date_month = ?", conn, params=[inmonth])
			else:
				print('Beginning SQL query for all posts between ' + str(mintime) + ' and ' + str(maxtime))
				df = pd.read_sql_query("SELECT timestamp, comment, title, num, date_full FROM pol_content WHERE timestamp > ? AND timestamp < ?;", conn, params=[mintime, maxtime])
		#look for string in subject
		elif stringintitle == True:
			if querystring2 != '':
				print('Beginning SQL query for "' + querystring + '" and "' + querystring2 + '" in post body')
				df = pd.read_sql_query("SELECT timestamp, title, comment, num, date_full FROM pol_content WHERE ((lower(title) LIKE ?) OR (lower(title) LIKE ?));", conn, params=['%' + querystring + '%', '%' + querystring2 + '%'])
			else:
				print('Beginning SQL query for "' + querystring + '" in post body')
				df = pd.read_sql_query("SELECT timestamp, comment, title, num, date_full FROM pol_content WHERE lower(title) LIKE ?;", conn, params=['%' + querystring + '%'])
		#look for sting in comment body (default)
		else:
			if querystring2 != '':
				print('Beginning SQL query for "' + querystring + '" and "' + querystring2 + '" in post body')
				df = pd.read_sql_query("SELECT timestamp, title, comment, num, date_full FROM pol_content WHERE ((lower(comment) LIKE ?) AND (lower(comment) LIKE ?));", conn, params=['%' + querystring + '%', '%' + querystring2 + '%'])
			else:
				if inmonth != '':
					print('Beginning SQL query for "' + querystring + '" in post body in ' + inmonth)
					df = pd.read_sql_query("SELECT timestamp, comment, title, num, date_full, date_month FROM pol_content WHERE lower(comment) LIKE ? AND date_month = ?;", conn, params=['%' + querystring + '%', inmonth])
				else:
					print('Beginning SQL query for "' + querystring + '" in post body')
					df = pd.read_sql_query("SELECT timestamp, comment, title, num, date_full FROM pol_content WHERE lower(comment) LIKE ?;", conn, params=['%' + querystring + '%'])

		# print('Writing results to csv')
		if '/' in querystring:
			querystring = re.sub(r'/', '', querystring)
		else:
			querystring = querystring
		df.to_csv('substring_mentions/mentions_' + querystring + '.csv')
	else:
		df = pd.read_csv('substring_mentions/mentions_trump/trump_threads/trump_threads_15percent_30min.csv', encoding='utf-8')

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
				words_stemmed = similarities.getTokens(li_str_timeseparated, li_stringdates, similaritytype, stems=False)
				similarities.getDocSimilarity(li_str_timeseparated, words_stemmed, li_stringdates, querystring)
			elif similaritytype == 'words':
				words_stemmed = similarities.getTokens(li_str_full, li_stringdates, similaritytype)
				similarities.getWordSimilarity(words_stemmed)

	if histogram == True:
		createHistogram.createHistogram(df, querystring=querystring, timeformat=inputtime, includenormalised=True)
	df_return = pd.DataFrame()
	df_return['comment'] = df['comment']
	#print(df)
	return df

def writeToText(inputdf, querystring, currenttime):

	directory = 'substring_mentions/longstring_' + querystring.replace(' ', '-')
	if not os.path.exists(directory):
		os.makedirs(directory)
	txtfile = open(directory + '/longstring_' + querystring + '_' + currenttime + '.txt', 'w', encoding='utf-8')
	str_keyword = ''
	li_str = []
	for item in inputdf['comments']:
		item = item.lower()
		regex = re.compile("[^a-zA-Z \.\,\-\n]")		#excludes numbers, might have to revise this
		item = regex.sub("", item)
		txtfile.write("%s" % item)
		str_keyword = str_keyword + item
		li_str.append(item)
	return str_keyword, li_str

def getTimeSeparatedCsvs(df='', timesep='months'):
	df_all = pd.read_csv(df, encoding='utf-8')
	if timesep == 'months':
		for index, tpl_time in enumerate(li_times_months):
			print('working on ' + li_labels_months[index])
			df_timesep = pd.DataFrame()
			df_timesep = df_all[df_all['timestamp'].between(tpl_time[0], tpl_time[1])]
			df_timesep.to_csv('substring_mentions/mentions_trump/months/mentions_trump_' + li_labels_months[index] + '.csv', encoding='utf-8')
	elif timesep == 'weeks':
		for index, tpl_time in enumerate(li_times_weeks):
			timestamp = tpl_time[0]
			print(timestamp)
			timetup = datetime.fromtimestamp(tpl_time[0]).isocalendar()
			year = (timetup)[0]
			week = (timetup)[1]
			print('working on ' + str(week) + '-' + str(year))
			df_timesep = pd.DataFrame()
			df_timesep = df_all[df_all['timestamp'].between(tpl_time[0], tpl_time[1])]
			df_timesep.to_csv('substring_mentions/mentions_trump/weeks/mentions_trump_' + str(year) + '-' + str(week) + '.csv', encoding='utf-8')

def getTrumpThreads(querystring='', getdf=True, maketables=False, getMetaInfo=False):
	print('Connecting to database')
	conn = sqlite3.connect("../4plebs_pol_18_03_2018.db")

	print('Fetching all OPs with "trump"')
	if maketables:
		
		cursor = conn.cursor()
		cursor.execute("""
					CREATE TABLE trump_threads_tmp AS
					SELECT thread_num as thread_no, trump_count FROM
					(
						SELECT thread_num, trump_count FROM (
							SELECT thread_num, count(*) as trump_count FROM
							(
								SELECT thread_num FROM poldatabase_18_03_2018
								WHERE (lower(comment) LIKE '%trump%' OR lower(title) LIKE '%trump%') AND timestamp > 1388534400
							)

							GROUP BY thread_num
							ORDER BY trump_count DESC
							)
						WHERE trump_count >= 1
						)
					;""")
		cursor=conn.cursor()
		cursor.execute("""
					CREATE TABLE trump_threads AS
					SELECT threadno, thread_count, trump_count, (trump_count*1.0) / (thread_count*1.0) as trump_density FROM (
						SELECT threadno, thread_count, trump_count FROM
							(
							SELECT threadno, trump_count, count(*) as thread_count
							FROM trump_threads
							INNER JOIN poldatabase_18_03_2018 ON trump_threads.threadno = poldatabase_18_03_2018.thread_num
							GROUP BY threadno
							ORDER BY thread_count
							)
						);""")

	if getdf:
		df_trumpthreads = pd.read_sql_query("""
					SELECT thread_num, num, op, timestamp, title, comment, timestamp_expired, media_filename, media_hash, name, trip, sticky, poster_hash, poster_country, date_month, date_week FROM pol_content
					WHERE thread_num IN (
        				SELECT trump_threads.threadno FROM trump_threads
        				WHERE (trump_threads.trump_density >= 0.15 AND trump_threads.thread_count >= 30)
					)
					""",conn)
		return df_trumpthreads
	if getMetaInfo:
		df_trumpthreads_meta = pd.read_sql_query("""
					SELECT * FROM trump_threads
					""",conn)
		print(df_trumpthreads_meta[:50])

