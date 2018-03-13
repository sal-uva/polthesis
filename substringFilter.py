
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

def substringFilter(inputstring, histogram = False, inputtime = 'months'):
	querystring = inputstring.lower()
	print('Connecting to database')
	conn = sqlite3.connect("../4plebs_pol_withheaders.db")

	print('Beginning SQL query for "' + querystring + '"')
	df = pd.read_sql_query("SELECT timestamp, comment FROM poldatabase WHERE lower(comment) LIKE ?;", conn, params=['%' + querystring + '%'])
	print('Writing results to csv')
	df.to_csv('test_output.csv')

	if histogram == True:
		createHistogram(df, querystring, inputtime)

def createHistogram(inputdf, querystring, inputtimeformat):
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
		print(type(timebuckets[1]))

	# plot it
	fig, ax = plt.subplots(1,1)
	ax.hist(mpl_dates, bins=timebuckets, align="left", color='red', ec="k")

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

result = substringFilter('richard spencer', histogram = True, inputtime='months')	#returns tuple with df and input string

print('finished')