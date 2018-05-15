import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import re
import os
from datetime import datetime, timedelta
from collections import OrderedDict
from matplotlib.ticker import ScalarFormatter

def createHistogram(df, querystring='', timeformat='months', includenormalised=False):
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
			li_timeticks.append(datetime(y, m + 1, 1).strftime("%m-%y"))
		#print(li_timeticks)
		dateformat = '%m-%y'
		mpl_dates = matplotlib.dates.epoch2num(li_timestamps)
		
		timebuckets = [datetime.strptime(i, "%m-%y") for i in li_timeticks]
		timebuckets = matplotlib.dates.date2num(timebuckets)

		#print(li_timeticks)
		count_timestamps = 0
		month = 0
		if includenormalised:
			di_totalcomment = {'11-13': 1, '12-13': 61519, '01-14': 1212183, '02-14': 1169314, '03-14': 1057428, '04-14': 1234152, '05-14': 1135162, '06-14': 1195313, '07-14': 1127383, '08-14': 1491844, '09-14': 1738433, '10-14': 1571584, '11-14': 1424278, '12-14': 1441101, '01-15': 909278, '02-15': 772111, '03-15': 890495, '04-15': 1197976, '05-15': 1300518, '06-15': 1381517, '07-15': 1392446, '08-15': 1597274, '09-15': 1903111, '10-15': 23000, '11-15': 26004, '12-15': 2344421, '01-16': 2592275, '02-16': 2925369, '03-16': 3111713, '04-16': 3736528, '05-16': 3048962, '06-16': 3131789, '07-16': 3642871, '08-16': 4314923, '09-16': 3618363, '10-16': 3759066, '11-16': 4418571, '12-16': 5515200, '01-17': 4187400, '02-17': 5191531, '03-17': 4368911, '04-17': 4386181, '05-17': 4428757, '06-17': 4374011, '07-17': 4020058, '08-17': 3752418, '09-17': 4087688, '10-17': 3703119, '11-17': 3931560, '12-17': 4122068,'01-18': 3584861, '02-18': 3624546, '03-18': 3468642}

			li_totalcomments = [1,61519,1212183,1169314,1057428,1234152,1135162,1195313,1127383,1491844,1738433,1571584,1424278,1441101,909278,772111,890495,1197976,1300518,1381517,1392446,1597274,1903111,2000023,2004026,2344421,2592275,2925369,3111713,3736528,3048962,3131789,3642871,4314923,3618363,3759066,4418571,5515200,4187400,5191531,4368911,4386181,4428757,4374011,4020058,3752418,4087688,3703119,3931560,4122068,3584861,3624546,3468642]

		#print(mpl_dates)
		#print(timebuckets)

	# plot it
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

def plotMultipleTrends(df1=None,df2=None,df3=None, query='', filename='', twoaxes=False):
	df_1 = df1.loc[1:].reset_index()
	print(df1)
	print(df_1)
	df_1['numbs'] = [x for x in range(len(df_1))]
	datelabels = [date for date in df_1['dateformatted']]
	df_2 = df2.loc[1:].reset_index()
	df_3 = df3.loc[1:].reset_index()

	fig = plt.figure(figsize=(12, 8))
	fig.set_dpi(100)
	ax = fig.add_subplot(111)

	if twoaxes:
		df_1.plot(ax=ax, y='percentage', label = 'trump', kind='line', legend=False, linewidth=2, color='orange');
		ax2 = ax.twinx()
		df_2.plot(ax=ax2, y='percentage',  label = 'god-emperor', kind='line', legend=False, linewidth=2, color='blue');
	else:
		df_1.plot(ax=ax, y='percentage', label = 'trump', kind='line', legend=False, linewidth=2, color='orange');
		df_2.plot(ax=ax, y='percentage',  label = 'nice', kind='line', legend=False, linewidth=2, color='blue');
		df_3.plot(ax=ax, y='percentage',  label = 'would', kind='line', legend=False, linewidth=2, color='green');
	ax.set_xticks(df_1['numbs'])
	ax.set_xticklabels(df_1['date'], rotation=90)

	lines, labels = ax.get_legend_handles_labels()
	plt.xlim([0,len(datelabels) -1])
	ax.set_ylim(bottom=0)
	ax.set_axisbelow(True)
	#ax2.set_axisbelow(True)
	ax.grid(color='#e5e5e5',linestyle='dashed', linewidth=.6)
	ax.set_ylabel('Percentage of total comments')
	plt.title('Percentage of 4chan/pol/ comments containing "' + query + '"')

	if twoaxes:
		lines2, labels2 = ax2.get_legend_handles_labels()
		print('do nothing')
		ax2.legend(lines + lines2, labels + labels2, loc='upper left')
	# 	lns = ln1 + ln2
	# 	labs = [l.get_label() for axes in ln1]
		ax2.set_ylim(bottom=0)
	# 	ax.legend(lns, labs, loc='upper left')
	else:
		ax.legend(loc='upper left')

	plt.savefig('../visualisations/substring_counts/' + filename + '_multiple.svg', dpi='figure')
	plt.savefig('../visualisations/substring_counts/' + filename + '_multiple.jpg', dpi='figure')
	#plt.show()

def createHistoFromTfidf(df='', li_words=''):
	df = df[df['word'].isin([li_words])]
	df = df.transpose()
	print(df)
	df['numbs'] = [x for x in range(len(df))]
	datelabels = [date for date in df['dateformatted']]

	fig = plt.figure(figsize=(12, 8))
	fig.set_dpi(100)
	ax = fig.add_subplot(111)

	df.plot(ax=ax, y='percentage', label = 'trump', kind='line', legend=False, linewidth=2, color='orange');
	
	ax.set_xticks(df['numbs'])
	ax.set_xticklabels(df['date'], rotation=90)

	lines, labels = ax.get_legend_handles_labels()
	plt.xlim([0,len(datelabels) -1])
	ax.set_ylim(bottom=0)
	ax.set_axisbelow(True)
	#ax2.set_axisbelow(True)
	ax.grid(color='#e5e5e5',linestyle='dashed', linewidth=.6)
	ax.set_ylabel('Percentage of total comments')
	plt.title('Percentage of 4chan/pol/ comments containing "' + query + '"')

	if twoaxes:
		lines2, labels2 = ax2.get_legend_handles_labels()
		ax2.legend(lines + lines2, labels + labels2, loc='upper left')
		ax2.set_ylim(bottom=0)
	else:
		ax.legend(loc='upper left')

	plt.savefig('tfidf/' + filename + '_trump_tfidf.svg', dpi='figure')
	plt.savefig('tfidf/' + filename + '_trump_tfidf.jpg', dpi='figure')
	#plt.show()

def createThreadMetaHisto(df=''):
	print('Creating bar chart for thread meta data')
	#df = df.sort_values(by='amount_of_posts', ascending=True)
	print(df.head())
	li_labels = []
	for count in df['amount_of_posts']:
		if count == '500':
			count = '500+'
			li_labels.append(count)
		else:
			li_labels.append(count)
	print(li_labels)

	fig = plt.figure(figsize=(12, 8))
	fig.set_dpi(100)
	ax = fig.add_subplot(111)
	ax2 = ax.twinx()
	ax3 = ax.twinx()
	kwarg = {'position': 1}
	df.plot(ax=ax, x='amount_of_posts', y='occurrances', kind='bar', label='Threads containing "trump"', position=0.0, legend=False, width=.9, color='#52b6dd');
	df.plot(ax=ax2, x='amount_of_posts', y='averagetrumps', kind='line', label='Trump count: Average posts with "trump" per thread', legend=False, linewidth=1.2, color='red');
	df.plot(ax=ax3, x='amount_of_posts', y='averagetrumpdensity', kind='line', label='Trump density: Percentage of total thread posts with "trump"', legend=False, linewidth=1.2, color='orange');
	plt.title('All threads on 4chan/pol/ containing "trump", separated by thread length')

	ax2.set_ylim(bottom=0, top=25)
	ax3.set_ylim(bottom=0, top=25)

	ax.grid(color='#e5e5e5',linestyle='dashed', linewidth=.6)

	ax.set_xticklabels(li_labels)
	#legend
	# lines, labels = ax.get_legend_handles_labels()
	# lines2, labels2 = ax2.get_legend_handles_labels()
	# lines3, labels3 = ax3.get_legend_handles_labels()
	# ax2.legend(lines + lines2 + lines3, labels + labels2 + labels, loc='upper right')
	ax.set_xlabel("Amount of posts in thread")
	ax.set_ylabel('Threads having 1> post(s) containing "trump"', color='#52b6dd')
	ax2.set_ylabel('Posts containing "trump", average per thread', color='red')
	ax3.set_ylabel('Percentage of posts containing "trump", per thread', color='orange')
	ax2.yaxis.set_label_coords(1.06,0.5)
	
	plt.show()

def getThreadMetaInfo(df=''):
	df['thread_count'] = [int(string) for string in df['thread_count']]
	print(len(df))
	df = df.sort_values(by='thread_count', ascending=True)
	print(df.head())
	print(df['thread_count'])

	di_threadlengths = {}
	di_average_trumps = {}
	di_average_trumpdensity = {}


	count_500 = 0
	li_500_trumps = []
	li_500_trumpdensities = []

	for index, count in enumerate(df['thread_count']):
		mod_count = count - (count % 10)
		str_mod_count = str(mod_count)
		if mod_count >= 500:
			count_500 += 1
			li_500_trumps.append(df['trump_count'][index])
			li_500_trumpdensities.append(df['trump_density'][index])
		elif str_mod_count in di_threadlengths:
			di_threadlengths[str_mod_count] += 1
			di_average_trumps[str_mod_count].append(df['trump_count'][index])
			di_average_trumpdensity[str_mod_count].append(df['trump_density'][index])
		else:
			di_threadlengths[str_mod_count] = 1
			print(str_mod_count)
			di_average_trumps[str_mod_count] = []
			di_average_trumpdensity[str_mod_count] = []
			di_average_trumps[str_mod_count].append(df['trump_count'][index])
			di_average_trumpdensity[str_mod_count].append(df['trump_density'][index])

	di_threadlengths['500'] = count_500
	di_average_trumps['500'] = li_500_trumps
	di_average_trumpdensity['500'] = li_500_trumpdensities

	#calculate the average trump count and trump density per length of thread ('do longer threads contain more Trumps?')
	averagetrumps = 0
	li_averagetrumps = []
	for key, value in di_average_trumps.items():
		for av in value:
			#print(av)
			averagetrumps += av
		averagetrumps = (averagetrumps / len(value))
		#print(averagetrumps)
		li_averagetrumps.append(averagetrumps)
	trumpdensities = 0
	li_trumpdensities = []
	for key, value in di_average_trumpdensity.items():
		for av in value:
			#print(av)
			trumpdensities += av
		#print(len(value), trumpdensities)
		trumpdensities = (trumpdensities / len(value)) * 100
		#print(trumpdensities)
		li_trumpdensities.append(trumpdensities)

	df_plot = pd.DataFrame.from_dict(di_threadlengths, orient='index')
	df_plot.reset_index(level=0, inplace=True)
	df_plot.columns = ['amount_of_posts','occurrances']
	df_plot['averagetrumps'] = li_averagetrumps
	df_plot['averagetrumpdensity'] = li_trumpdensities
	print(df_plot.head())
	return df_plot

df = pd.read_csv('substring_mentions/mentions_trump/trump_threads/trump_threads.csv')
df = df.groupby(['date_month']).agg(['count']
