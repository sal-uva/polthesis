
import sqlite3
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import OrderedDict

#HEADERS: num, subnum, thread_num, op, timestamp, timestamp_expired, preview_orig,
# preview_w, preview_h, media_filename, media_w, media_h, media_size,
# media_hash, media_orig, spoiler, deleted, capcode, email, name, trip,
# title, comment, sticky, locked, poster_hash, poster_country, exif

#all comments:
# 2013-12	61519
# 2014-01	1212183
# 2014-02	1169314
# 2014-03	1057428
# 2014-04	1234152
# 2014-05	1135162
# 2014-06	1195313
# 2014-07	1127383
# 2014-08	1491844
# 2014-09	1738433
# 2014-10	1571584
# 2014-11	1424278
# 2014-12	1441101
# 2015-01	909278
# 2015-02	772111
# 2015-03	890495
# 2015-04	1197976
# 2015-05	1300518
# 2015-06	1381517
# 2015-07	1392446
# 2015-08	1597274
# 2015-09	1903111
# 2015-10	2000023
# 2015-11	2004026
# 2015-12	2344421
# 2016-01	2592275
# 2016-02	2925369
# 2016-03	3111713
# 2016-04	3736528
# 2016-05	3048962
# 2016-06	3131789
# 2016-07	3642871
# 2016-08	4314923
# 2016-09	3618363
# 2016-10	3759066
# 2016-11	4418571
# 2016-12	5515200
# 2017-01	4187400
# 2017-02	5191531
# 2017-03	4368911
# 2017-04	4386181
# 2017-05	4428757
# 2017-06	4374011
# 2017-07	4020058
# 2017-08	3752418
# 2017-09	4087688
# 2017-10	3703119
# 2017-11	3931560
# 2017-12	4122068
# 2018-01	3584861
# 2018-02	3624546
# 2018-03	3468642


def substringFilter(inputstring, histogram = False, inputtime = 'months', normalised=False):
	querystring = inputstring.lower()
	print('Connecting to database')
	conn = sqlite3.connect("../4plebs_pol_18_03_2018.db")

	print('Beginning SQL query for "' + querystring + '"')
	df = pd.read_sql_query("SELECT timestamp, comment FROM poldatabase_18_03_2018 WHERE lower(comment) LIKE ?;", conn, params=['%' + querystring + '%'])
	print('Writing results to csv')
	df.to_csv('test_output.csv')

	if histogram == True:
		createHistogram(df, querystring, inputtime, normalised)

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
		for tot_m in range(total_months(start)-1, total_months(end)):
			y, m = divmod(tot_m, 12)
			li_timeticks.append(datetime(y, m+1, 1).strftime("%m-%y"))
		print(li_timeticks)
		dateformat = '%m-%y'
		mpl_dates = matplotlib.dates.epoch2num(li_timestamps)
		
		timebuckets = [datetime.strptime(i, "%m-%y") for i in li_timeticks]
		timebuckets = matplotlib.dates.date2num(timebuckets)

		print(li_timeticks)
		count_timestamps = 0
		month = 0
		mpl_normaldates = []
		if normalised:
			di_totalcomment = {'2013-11': 0, '2013-12': 61519, '2014-01': 1212183, '2014-02': 1169314, '2014-03': 1057428, '2014-04': 1234152, '2014-05': 1135162, '2014-06': 1195313, '2014-07': 1127383, '2014-08': 1491844, '2014-09': 1738433, '2014-10': 1571584, '2014-11': 1424278, '2014-12': 1441101, '2015-01': 909278, '2015-02': 772111, '2015-03': 890495, '2015-04': 1197976, '2015-05': 1300518, '2015-06': 1381517, '2015-07': 1392446, '2015-08': 1597274, '2015-09': 1903111, '2015-10': 2000023, '2015-11': 2004026, '2015-12': 2344421, '2016-01': 2592275, '2016-02': 2925369, '2016-03': 3111713, '2016-04': 3736528, '2016-05': 3048962, '2016-06': 3131789, '2016-07': 3642871, '2016-08': 4314923, '2016-09': 3618363, '2016-10': 3759066, '2016-11': 4418571, '2016-12': 5515200, '2017-01': 4187400, '2017-02': 5191531, '2017-03': 4368911, '2017-04': 4386181, '2017-05': 4428757, '2017-06': 4374011, '2017-07': 4020058, '2017-08': 3752418, '2017-09': 4087688, '2017-10': 3703119, '2017-11': 3931560, '2017-12': 4122068, '2018-01': 3584861, '2018-02': 3624546, '2018-03': 3468642}

			li_totalcomments = [0,61519,1212183,1169314,1057428,1234152,1135162,1195313,1127383,1491844,1738433,1571584,1424278,1441101,909278,772111,890495,1197976,1300518,1381517,1392446,1597274,1903111,2000023,2004026,2344421,2592275,2925369,3111713,3736528,3048962,3131789,3642871,4314923,3618363,3759066,4418571,5515200,4187400,5191531,4368911,4386181,4428757,4374011,4020058,3752418,4087688,3703119,3931560,4122068,3584861,3624546,3468642]

			for timestamp in mpl_dates:
				if(str(datetime.fromtimestamp(timestamp).strftime("%b-%Y"))) == li_timeticks[month]:
					count_timestamps = count_timestamps + 1
				else:
					normalised_mentions = (count_timestamps / (li_totalcomments[month] + 1)) * 100
					mpl_normaldates.append(normalised_mentions)
					month = month + 1
			mpl_dates = mpl_normaldates
		print(type(timebuckets[1]))
		print(mpl_dates)
		print(timebuckets)

	# plot it!
	fig, ax = plt.subplots(1,1)
	ax.hist(mpl_dates, bins=timebuckets, align="left", color='red', ec="k")
	histo = ax.hist(mpl_dates, bins=timebuckets, align="left", color='red', ec="k")
	print(histo)
	print(histo[0])
	li_counts= []
	for counts in histo[0]:
		li_counts.append(counts)
	print('Writing results to textfile')
	write_handle = open('deusvultcounts.txt',"w")
	write_handle.write(str(li_counts))
	write_handle.close()

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
	plt.show()

result = substringFilter('mary poppins', histogram = True, inputtime='months', normalised=True)	#returns tuple with df and input string

print('finished')