
import sqlite3
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import re
from nltk.corpus import stopwords
from scipy.interpolate import spline
from datetime import datetime, timedelta
from collections import OrderedDict
from sklearn.feature_extraction.text import TfidfVectorizer

#HEADERS: num, subnum, thread_num, op, timestamp, timestamp_expired, preview_orig,
# preview_w, preview_h, media_filename, media_w, media_h, media_size,
# media_hash, media_orig, spoiler, deleted, capcode, email, name, trip,
# title, comment, sticky, locked, poster_hash, poster_country, exif

def substringFilter(inputstring, histogram = False, inputtime = 'months', normalised=False, writetext=False, textsimilarity = False):
	querystring = inputstring.lower()
	print('Connecting to database')
	conn = sqlite3.connect("../4plebs_pol_test_database.db")

	print('Beginning SQL query for "' + querystring + '"')
	df = pd.read_sql_query("SELECT timestamp, comment FROM poldatabase WHERE lower(comment) LIKE ?;", conn, params=['%' + querystring + '%'])
	print('Writing results to csv')
	df.to_csv('substring_mentions/mentions_' + querystring + '.csv')

	if writetext == True:
		df_parsed = pd.DataFrame(columns=['comments','time'])
		df_parsed['comments'] = df['comment']
		#note: make day seperable later
		df_parsed['time'] = [datetime.strftime(datetime.fromtimestamp(i), "%m-%Y") for i in df['timestamp']]
		df_parsed['comments'] = [re.sub(r'>', '', z) for z in df_parsed['comments']]
		
		print(df_parsed['comments'])
		#write text file for separate months
		currenttime = df_parsed['time'][1]
		oldindex = 1

		li_keystrings = []

		#create text files for each month
		for index, distincttime in enumerate(df_parsed['time']):
			if distincttime != currenttime or index == (len(df_parsed['time']) - 1):
				print(currenttime, distincttime)
				
				df_sliced = df_parsed[oldindex:index]
				print(df_sliced)
				string = writeToText(df_sliced, querystring, currenttime)
				li_keystrings.append(string)
				oldindex = index + 1
				currenttime = distincttime				

	# FOR DEBUGGING PURPOSES:
	#df = pd.read_csv('substring_mentions/mentions_alt-left.csv')
	
	if textsimilarity == True:
		#do some cleanup: only alphabetic characters, no stopwords
		for string in li_keystrings:
			regex = re.compile('[^a-zA-Z]')
			regex.sub('', string)
			wordlist = re.sub("[^\w]", " ", string).split()
			wordlist = [word for word in wordlist if word not in set(stopwords.words('english'))]
			string = ' '.join(wordlist)
			#print(string)
		
		print('Calculating cosine differences')
		vect = TfidfVectorizer(min_df=1)
		tfidf = vect.fit_transform(li_keystrings)
		similarityvector = (tfidf * tfidf.T).A
		print(similarityvector)
		similarityfile = open('substring_mentions/tfidf_' + querystring + '.txt', 'w')
		similarityfile.write(str(similarityvector))

	if histogram == True:
		createHistogram(df, querystring, inputtime, normalised)

def writeToText(inputdf, querystring, currenttime):
	txtfile = open('substring_mentions/longstring_' + querystring + '_' + currenttime + '.txt', 'w', encoding='utf-8')
	str_keyword = ''
	for item in inputdf['comments']:
		item = item.lower()
		regex = re.compile("[^a-zA-Z \.\n]")		#excludes numbers, might have to revise this
		item = regex.sub("", item)
		txtfile.write("%s" % item)
		str_keyword = str_keyword + item
	return str_keyword

def createHistogram(inputdf, querystring, inputtimeformat, normalised):
	df = inputdf
	timeformat = inputtimeformat
	li_timestamps = df['timestamp'].values.tolist()

	li_timeticks = []

	dateformat = '%d-%m-%y'

	if timeformat == 'days':
		one_day = timedelta(days = 1)
		startdate = datetime.fromtimestamp(li_timestamps[0])
		enddate = datetime.fromtimestamp(li_timestamps[len(li_timestamps) - 1])
		delta =  enddate - startdate
		print(startdate)
		print(enddate)
		count_days = delta.days + 2
		print(count_days)
		for i in range((enddate-startdate).days + 1):
		    li_timeticks.append(startdate + (i) * one_day)
		dateformat = '%d-%m-%y'
		#convert UNIX timespamp
		mpl_dates = matplotlib.dates.epoch2num(li_timestamps)
		timebuckets = matplotlib.dates.date2num(li_timeticks)

	elif timeformat == 'months':
		#one_month = datetime.timedelta(month = 1)
		startdate = (datetime.fromtimestamp(li_timestamps[0])).strftime("%Y-%m-%d")
		enddate = (datetime.fromtimestamp(li_timestamps[len(li_timestamps) - 1])).strftime("%Y-%m-%d")
		dates = [str(startdate), str(enddate)]
		start, end = [datetime.strptime(_, "%Y-%m-%d") for _ in dates]
		total_months = lambda dt: dt.month + 12 * dt.year
		li_timeticks = []
		for tot_m in range(total_months(start) - 1, total_months(end)):
			y, m = divmod(tot_m, 12)
			li_timeticks.append(datetime(y, m+1, 1).strftime("%m-%y"))
		#print(li_timeticks)
		dateformat = '%m-%y'
		mpl_dates = matplotlib.dates.epoch2num(li_timestamps)
		
		timebuckets = [datetime.strptime(i, "%m-%y") for i in li_timeticks]
		timebuckets = matplotlib.dates.date2num(timebuckets)

		#print(li_timeticks)
		count_timestamps = 0
		month = 0
		if normalised:
			di_totalcomment = {'11-13': 1, '12-13': 61519, '01-14': 1212183, '02-14': 1169314, '03-14': 1057428, '04-14': 1234152, '05-14': 1135162, '06-14': 1195313, '07-14': 1127383, '08-14': 1491844, '09-14': 1738433, '10-14': 1571584, '11-14': 1424278, '12-14': 1441101, '01-15': 909278, '02-15': 772111, '03-15': 890495, '04-15': 1197976, '05-15': 1300518, '06-15': 1381517, '07-15': 1392446, '08-15': 1597274, '09-15': 1903111, '10-15': 23000, '11-15': 26004, '12-15': 2344421, '01-16': 2592275, '02-16': 2925369, '03-16': 3111713, '04-16': 3736528, '05-16': 3048962, '06-16': 3131789, '07-16': 3642871, '08-16': 4314923, '09-16': 3618363, '10-16': 3759066, '11-16': 4418571, '12-16': 5515200, '01-17': 4187400, '02-17': 5191531, '03-17': 4368911, '04-17': 4386181, '05-17': 4428757, '06-17': 4374011, '07-17': 4020058, '08-17': 3752418, '09-17': 4087688, '10-17': 3703119, '11-17': 3931560, '12-17': 4122068,'01-18': 3584861, '02-18': 3624546, '03-18': 3468642}

			li_totalcomments = [1,61519,1212183,1169314,1057428,1234152,1135162,1195313,1127383,1491844,1738433,1571584,1424278,1441101,909278,772111,890495,1197976,1300518,1381517,1392446,1597274,1903111,2000023,2004026,2344421,2592275,2925369,3111713,3736528,3048962,3131789,3642871,4314923,3618363,3759066,4418571,5515200,4187400,5191531,4368911,4386181,4428757,4374011,4020058,3752418,4087688,3703119,3931560,4122068,3584861,3624546,3468642]

		#print(mpl_dates)
		#print(timebuckets)

	# plot it!
	fig, ax = plt.subplots(1,1)
	ax.hist(mpl_dates, bins=timebuckets, align="left", color='red', ec="k")
	histo = ax.hist(mpl_dates, bins=timebuckets, align="left", color='red', ec="k")

	if timeformat == 'days':
		ax.xaxis.set_major_locator(matplotlib.dates.DayLocator())
		ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter(dateformat))
	elif timeformat == 'months':
		ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
		ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter(dateformat))

	#hide every N labels
	for label in ax.xaxis.get_ticklabels()[::2]:
		label.set_visible(False)

	#set title
	ax.set_title('4chan/pol/ comments containing "' + querystring + '", ' + str(startdate) + ' - ' + str(enddate))

	#rotate labels
	plt.gcf().autofmt_xdate()
	plt.savefig('../visualisationsold/' + querystring + '_full.svg')
	# plt.show(block=False)
	# time.sleep(2)
	plt.close()

	newli_timeticks = list(di_totalcomment.keys())
	#print(len(newli_timeticks))

	li_counts = []
	#print(ax.xaxis.get_ticklabels())
	li_axisticks = ax.xaxis.get_majorticklabels()
	li_axisticks = li_axisticks[:-3]
	li_axisticks = li_axisticks[3:]
	#print(li_axisticks)
	li_matchticks = []
	for text in li_axisticks:
		strtext = text.get_text()
		li_matchticks.append(strtext)
	print(len(li_matchticks))
	print('matching months: ' + str(li_matchticks))
	#print('histo data:')
	#print(histo)
	#print('histo len: ' + str(len(histo[0])))
	#print(len(li_matchticks))
	#print(li_matchticks)
	#print(newli_timeticks)
	#print(histo[0])
	#loop over each month
	histoindex = 0
	for month in range(53):
		occurance = False
		for registeredmonth in li_matchticks:
			#print(registeredmonth)
			if registeredmonth == newli_timeticks[month]:
				print('Month ' + str(histoindex) + ' found')
				li_counts.append(histo[0][histoindex])
				histoindex = histoindex + 1
				occurance = True
		if occurance == False:						#if the string did not occur, write 0
			print('no occurances this month')
			li_counts.append(0)
	
	print('li_counts length: ' + str(len(li_counts)))

	li_normalisedcounts = []
	for index, count in enumerate(li_counts):
		#print(li_totalcomments[index])
		count_normalised = (count / li_totalcomments[index]) * 100
		#print(count_normalised)
		li_normalisedcounts.append(count_normalised)

	li_datesformatted = []
	li_histodates = []
	for date in newli_timeticks:
		dateobject = datetime.strptime(date, '%m-%y')
		formatteddate = datetime.strftime(dateobject, '%Y-%b')
		histodate = datetime.strftime(dateobject, '%b %y')
		li_datesformatted.append(formatteddate)
		li_histodates.append(histodate)

	print('Writing results to csv')

	# only keep months that have full data
	del newli_timeticks[0]
	del li_datesformatted[0]
	del li_histodates[0]
	del li_normalisedcounts[0]
	del li_counts[0]
	del newli_timeticks[len(newli_timeticks) - 1]
	del li_histodates[len(li_histodates) - 1]
	del li_normalisedcounts[len(li_normalisedcounts) - 1]
	del li_counts[len(li_counts) - 1]
	del li_datesformatted[len(li_datesformatted) - 1]

	print(li_datesformatted)
	print(len(newli_timeticks))
	print(len(li_datesformatted))
	print(len(li_normalisedcounts))
	print(len(li_counts))

	finaldf = pd.DataFrame(columns=['date','dateformatted','count','percentage'])
	finaldf['date'] = newli_timeticks
	finaldf['dateformatted'] = li_datesformatted
	finaldf['percentage'] = li_normalisedcounts
	finaldf['count'] = li_counts
	finaldf.to_csv('substring_mentions/occurrances_' + querystring + '.csv', index=False)

	df2 = pd.DataFrame(index=li_histodates[1:], columns=['count','percentage'])
	df2['dates'] = li_histodates[1:]
	df2['count'] = li_counts[1:]
	df2['percentage'] = li_normalisedcounts[1:]
	plotNewGraph(df2, querystring)

def plotNewGraph(df, query):

	fig = plt.figure(figsize=(12, 8))
	fig.set_dpi(100)
	ax1 = fig.add_subplot(111)
	ax2 = ax1.twinx()

	df.plot(ax=ax1, y='count', kind='bar', legend=False, width=.9, color='#52b6dd');
	df.plot(ax=ax2, y='percentage', legend=False, kind='line', linewidth=2, color='#d12d04');
	ax1.set_axisbelow(True)
	ax1.set_xticklabels(df['dates'])
	ax1.grid(color='#e5e5e5',linestyle='dashed', linewidth=.6)
	ax1.set_ylabel('Absolute amount', color='#52b6dd')
	ax2.set_ylabel('Percentage of total comments', color='#d12d04')
	ax2.set_ylim(bottom=0)
	plt.title('Amount of 4chan/pol/ comments containing "' + query + '"')

	plt.savefig('../visualisations/substring_counts/' + query + '.svg', dpi='figure')
	plt.savefig('../visualisations/substring_counts/' + query + '.jpg', dpi='figure')

li_querywords = ['skyrim']

for word in li_querywords:
	result = substringFilter(word, histogram = True, inputtime='months', normalised=True, writetext=True, textsimilarity = True)	#returns tuple with df and input string
print('finished')