import sqlite3
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import datetime

#HEADERS: num, subnum, thread_num, op, timestamp, timestamp_expired, preview_orig,
# preview_w, preview_h, media_filename, media_w, media_h, media_size,
# media_hash, media_orig, spoiler, deleted, capcode, email, name, trip,
# title, comment, sticky, locked, poster_hash, poster_country, exif

def substringFilter(inputstring, histogram=False):
	querystring = inputstring.lower()
	conn = sqlite3.connect("../4plebs_pol_test_database.db")
	df = pd.read_sql_query("SELECT timestamp, comment FROM 'polsample' WHERE lower(comment) LIKE ?;", conn, params=['%' + querystring + '%'])
	print(df)
	df.to_csv('test_output.csv')
	createHistogram(df, querystring)

def createHistogram(inputdf, querystring):
	df = inputdf
	li_timestamps = df['timestamp'].values.tolist()

	# set default sans-serif font
	matplotlib.rcParams['font.sans-serif'] = "Courier"
	# state ALWAYS use sans-serif fonts
	matplotlib.rcParams['font.family'] = "sans-serif"

	one_day = datetime.timedelta(days = 1)
	startdate = datetime.datetime.fromtimestamp(li_timestamps[0])
	enddate = datetime.datetime.fromtimestamp(li_timestamps[len(li_timestamps) - 1])
	delta =  enddate - startdate

	print(startdate)
	print(enddate)
	count_days = delta.days + 2
	print(count_days)
	li_days = []
	for i in range((enddate-startdate).days+1):  
	    li_days.append(startdate + (i)*one_day)

	#convert UNIX timespamp
	mpl_dates = matplotlib.dates.epoch2num(li_timestamps)
	numdays = matplotlib.dates.date2num(li_days)

	# plot it
	fig, ax = plt.subplots(1,1)
	ax.hist(mpl_dates, bins=numdays, align="left", color='red', ec="k")
	ax.xaxis.set_major_locator(matplotlib.dates.DayLocator())
	ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%d-%m-%y'))

	#hide every N labels
	for label in ax.xaxis.get_ticklabels()[::2]:
	    label.set_visible(False)

	#set title
	ax.set_title('4chan/pol/ comments containing "' + querystring + '", ' + str(startdate)[:-9] + ' - ' + str(enddate)[:-9])

	#rotate labels
	plt.gcf().autofmt_xdate()
	plt.show()

result = substringFilter('donald trump', histogram=True)	#params: query and generate histogram

print('finished')