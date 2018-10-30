import sys
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import re
import os
from datetime import date, datetime, timedelta
from collections import OrderedDict
from matplotlib.ticker import ScalarFormatter

def createHistogram(df, querystring='', time_column='timespan', time_format='months', include_normalised=False):
	
	li_dates = df[time_column].values.tolist()
	li_timeticks = []

	dateformat = '%d-%m-%y'

	df_histo = pd.DataFrame()

	if time_format == 'months':
		df['date_histo'] = [date[:7] for date in df[time_column]]
		df = df.groupby(by=['date_histo']).agg(['count'])
		# print(df)

	elif time_format == 'days':
		df['date_histo'] = [date[:10] for date in df[time_column]]
		df = df.groupby(by=['date_histo']).agg(['count'])
		# print(df)

	# Create new list of all dates between start and end date
	# Sometimes one date has zero counts, and gets skipped by matplotlib
	li_dates = []
	li_check_dates = []
	if time_format == 'months':
		d1 = datetime.strptime(df.index[0], "%Y-%m").date()  			# start date
		d2 = datetime.strptime(df.index[len(df) - 1], "%Y-%m").date()	# end date
		delta = d2 - d1         										# timedelta
		for i in range(delta.days + 1):
			date = d1 + timedelta(days=i)
			date = str(date)[:7]
			if date not in li_dates:
				li_dates.append(date)
		
	if time_format == 'days':
		d1 = datetime.strptime(df.index[0], "%Y-%m-%d").date()			# start date
		d2 = datetime.strptime(df.index[len(df) - 1], "%Y-%m-%d").date()# end date
		# print(d1, d2)
		delta = d2 - d1         										# timedelta
		for i in range(delta.days + 1):
			date_object = d1 + timedelta(days=i)
			str_date = date_object.strftime('%Y-%m-%d')
			li_dates.append(date_object)
			li_check_dates.append(str_date)

	# Create list of counts. 0 if it does not appears in previous DataFrame
	li_counts = [0 for i in range(len(li_dates))]
	li_index_dates = df.index.values.tolist()
	for index, indate in enumerate(li_check_dates):
		# print(indate)
		if indate in li_index_dates and df.loc[indate][1] > 0:
			li_counts[index] = df.loc[indate][1]
		else:
			li_counts[index] = 0

	df_histo['date'] = li_dates
	df_histo['count'] = li_counts

	#create list of average counts
	if include_normalised:
		li_av_count = []
		for i in range(len(df_histo)):
			av_count = (df_histo['id'][i] / di_totalcomment[df_histo['date'][i]]) * 100
			li_av_count.append(av_count)

		df_histo['av_count'] = li_av_count

	if not os.path.exists('output/'):
		os.makedirs('output/')

	print(df_histo)

	# Safe the metadata
	df_histo.to_csv('output/histogram_data_' + querystring + '.csv', index=False)

	# Plot the graph!
	fig, ax = plt.subplots(1,1)
	fig = plt.figure(figsize=(12, 8))
	fig.set_dpi(100)
	ax = fig.add_subplot(111)

	#ax2 = ax.twinx()
	if time_format == 'days':
		ax.xaxis.set_major_locator(matplotlib.dates.DayLocator())
		ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter(dateformat))
	elif time_format == 'months':
		ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
		ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter(dateformat))
	ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
	
	df_histo.plot(ax=ax, y='count', kind='bar', legend=False, width=.9, color='#52b6dd');
	#df_histo.plot(ax=ax2, y='av_count', legend=False, kind='line', linewidth=2, color='#d12d04');
	ax.set_axisbelow(True)
	# ax.set_xticks(xticks)
	ax.set_xticklabels(df_histo['date'], rotation='vertical')
	ax.grid(color='#e5e5e5',linestyle='dashed', linewidth=.6)
	ax.set_ylabel('Absolute amount', color='#52b6dd')
	#ax2.set_ylabel('Percentage of total comments', color='#d12d04')
	#ax2.set_ylim(bottom=0)
	plt.title('Posts containing "' + querystring + '"')

	plt.savefig('output/histogram_' + querystring + '.svg', dpi='figure',bbox_inches='tight')
	plt.savefig('output/histogram_' + querystring + '.png', dpi='figure',bbox_inches='tight')

	print('Done! Saved .csv of data and .png & .svg in folder \'output/\'')

# show manual if needed
if len(sys.argv) < 2:
	print()
	print("Creates a histogram of occurances of substring per day or month.")
	print("Saves the data in a csv file and visualises in .png and .svg.")
	print()
	print("Usage: python3 createHistogram.py [--source] [--output] [--timespan] [--timecolumn]")
	print()
	print("--source: the relative path to a csv file from 4CAT (e.g. 'data/datasheet.csv').")
	print("--string: the string that was filtered on in the csv. Used as plot title and output file.")
	print("--timespan: default 'days' - the separation of the histogram bars. Can be 'days' or 'months'.")
	print("--timecolumn: default 'timespan' - the csv column in which the time values are stored. Should start with format dd-mm-yyyy.")
	print()
	print("Example: python createHistogram.py --input=data/4cat-datasheet.csv --string=obama --timespan=months --timecolumn=months")
	print()
	sys.exit(1)

else:
	li_args = []
	source = ''
	output = ''
	querystring = ''
	time_column = 'timestamp'
	timespan = 'days'

	# Interpret command line arguments
	for arg in sys.argv:
		if arg[0:9] == "--source=":
			source = arg[9:len(arg)]
			li_args.append(source)
		elif arg[0:9] == "--string=":
			querystring = arg[9:len(arg)]
			li_args.append(querystring)
		elif arg[0:11] == "--timespan=":
			timespan = arg[11:len(arg)]
			li_args.append(timespan)
		elif arg[0:13] == "--timecolumn=":
			time_column = arg[13:len(arg)]
			li_args.append(time_column)

	if source == '' or not os.path.isfile(source):
		print("Please provide a valid input file like this: --source=data/datasheet.csv")
		sys.exit(1)
	elif querystring == '':
		print("Please provide what query string your input file is filtered on.")
		print("Used as plot title and output name.")
		sys.exit(1)
	else:
		df = pd.read_csv(source)
		createHistogram(df, querystring=querystring, time_format=timespan, time_column=time_column)