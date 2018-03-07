import sqlite3
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import datetime
import operator

# HEADERS: num, subnum, thread_num, op, timestamp, timestamp_expired, preview_orig,
# preview_w, preview_h, media_filename, media_w, media_h, media_size,
# media_hash, media_orig, spoiler, deleted, capcode, email, name, trip,
# title, comment, sticky, locked, poster_hash, poster_country, exif


def getTopURLs(piegraph = False, threshold=20):
	conn = sqlite3.connect("../4plebs_pol_test_database.db")

	urlSQLquery = "SELECT comment, thread_num, timestamp, poster_country FROM polsample WHERE (comment LIKE '%http://%' OR comment LIKE '%https://%' OR comment LIKE '%www.%');"
	df = pd.read_sql_query(urlSQLquery, conn)
	df['url'] = df['comment'].str.extract('([\/\/|www\.][0-9a-z\.]*\.[0-9a-z\.]+)')
	print(df['url'][:9])
	df.to_csv('top_urls.csv')

	li_urls = df['url'].values.tolist()

	di_all_urls = {}
	for url in li_urls:
		if url not in di_all_urls:
			di_all_urls[url] = 1
		else:
			di_all_urls[url] += 1
	print(di_all_urls)

	most_used_url = max(di_all_urls.items(), key=operator.itemgetter(1))[0]
	di_pie_urls = {}
	di_pie_urls['other'] = 0
	plotthreshold = di_all_urls[most_used_url] / threshold

	for key, value in di_all_urls.items():
		formatted_url = str(key)[1:]
		if '..' not in formatted_url:	#bugfix
			if value < plotthreshold: 		#if the URL is used 10 times or less than the most popular URL
				di_pie_urls['other'] += 1
			else:
				di_pie_urls[formatted_url] = value

	createPieGraph(di_pie_urls)

def createPieGraph(dictionary):
	di = dictionary
	pie_values = di.values()
	pie_labels = di.keys()
	
	fig, ax = plt.subplots()

	# Draw the pie chart
	ax.pie([float(v) for v in di.values()], labels=pie_labels, autopct='%1.2f', startangle=0, pctdistance = 0.9, labeldistance=1.2)	

	# Aspect ratio - equal means pie is a circle
	ax.axis('equal')
	ax.set_title('Top 4chan/pol/ URLs, 29-11-2013 to 21-01-2014')

	plt.show()

getTopURLs(piegraph=True, threshold=70)

