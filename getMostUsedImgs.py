import sqlite3
import pandas as pd
import io
import os
import urllib.request, json
import time
from PIL import Image

user_agent = 'Mozilla/5.0 (Windows NT 6.1; Win64; x64)'
headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
   'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
   'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
   'Accept-Encoding': 'none',
   'Accept-Language': 'en-US,en;q=0.8',
   'Connection': 'keep-alive'}

def getMostUsedImgs(querystring='', hash_threshold=0, loadcsv='', separate_month=False, stringintitle=False):
	"""
	Downloads the most used images based on their hashes that appear alongside a string.

	querystring:	str,	what string to filter on
	hash_treshold:	int,	a threshold for the minimum amount of hash appearances 
	loadcsv:		str,	can point to an already existing 4plebs csv to use instead of querying the database
	separate_month:	bool,	can be used to get the most popular images per month
	stringintitle:	bool,	whether to check for the string in the post title

	"""

	querystring = querystring.lower()
	if querystring == '':
		return 'Please provide a string'

	# If a csv is loaded, load csv, else check the database
	if loadcsv != '':
		print('Reading csv file')
		df = pd.read_csv(loadcsv, encoding='utf-8')
	else:
		print('Connecting to database')
		conn = sqlite3.connect("../4plebs_pol_18_03_2018.db")

		print('Beginning SQL query for "' + querystring + '"')
		if querystring == 'all':
			querystring = querystring + '-' + str(datetime.strftime(datetime.fromtimestamp(mintime), "%m-%Y"))
			df = pd.read_sql_query("SELECT timestamp, media_hash FROM pol_content WHERE timestamp > ? AND timestamp < ?;", conn, params=[mintime, maxtime])
		
		# Look for string in subject
		elif stringintitle == False:
			df = pd.read_sql_query("SELECT timestamp, media_hash FROM pol_content WHERE lower(comment) LIKE ?;", conn, params=['%' + querystring + '%'])
		
		# Look for sting in comment body (default)
		else:
			df = pd.read_sql_query("SELECT timestamp, media_hash FROM poldatabase WHERE lower(title) LIKE ?;", conn, params=['%' + querystring + '%'])
	
	if separate_month == True:
		li_months = df.groupby('date_month')
		li_months = list(df['date_months'])
		for month in li_months:
			getImgs(df=df[df['date_month'] == month], querystring=querystring, hash_threshold=hash_threshold, timesep=True)
	else:
		getImgs(df=df, querystring=querystring, hash_threshold=hash_threshold, timesep=False)

def getImgs(df, querystring, hash_threshold, timesep=False, resize=False):
	li_failedimgs = []

	# Create df with most used hashes
	df['occurrances'] = [1] * len(df)
	df = df.groupby('media_hash').agg({'occurrances': len})
	df = df.sort_values(by=['occurrances'], ascending=False)

	df['hash'] = df.index.values
	df.reset_index(level=0, inplace=True)
	print(df.head())

	# Make folder to store images in
	if os.path.exists('mostused_img/'+ querystring +'/') == False:
		os.makedirs('mostused_img/'+ querystring +'/')
	
	# Download image every 8 seconds
	for index, count_image in enumerate(df['occurrances']):
		imghash = df['hash'][index]
		if count_image >= hash_threshold:
			print('above threshold')
			url = 'http://archive.4plebs.org/_/api/chan/search/?boards=pol&image=' + imghash
			print(url)
			request = urllib.request.Request(url, headers=headers)
			
			#check if the thread is still active on 4plebs
			try:
				response = urllib.request.urlopen(request)

			#some threads get deleted and return a 404
			except urllib.error.HTTPError as httperror:
				print('HTTP error when requesting thread')
				print('Reason:', httperror.code)
				pass
			else:
				data = response.read().decode('utf-8', 'ignore')
				postdata = json.loads(data)

				if 'error' not in postdata:
					# loop through posts with image hash until image is downloaded (some are invalid)
					print(len(postdata['0']['posts']))
					print('Trying to fetch image')
					postmedia = postdata['0']['posts'][0]['media']
					print(postmedia)

					if 'media_link' in postmedia:
						print('image found')
						img_link = postmedia['media_link']
						thumb_link = postmedia['thumb_link']
						print(img_link)
						img_request = urllib.request.Request(img_link, headers=headers)

						try:
							img_response = urllib.request.urlopen(img_request)
						except urllib.error.HTTPError as httperror:
							print('HTTP error when requesting thread')
							print('Reason:', httperror.code)
							time.sleep(7)
							#if the image can't be found, try to get the thumbnail
							getThumbImg(thumb_link, imghash, count_image, querystring, timesep=timesep)
							pass
						
						else:
								imagefile = io.BytesIO(img_response.read())
								if '.webm' not in imagefile:
									image = Image.open(imagefile)

									if resize:
										imagesize = image.size
										if imagesize[0] > 800 or imagesize[1] > 800:
											print('Resizing...')
											image.thumbnail(size)

									imghash = imghash.replace('/','slash')
									image.save('mostused_img/' + querystring + '/' + querystring + '_' + str(count_image) + '_' + imghash + '.' + str(image.format))
					else:
						print('no media found')
					print('sleeping...')
					time.sleep(8)
				
				else:
					li_failedimgs.append(imghash)
					print('invalid image')
					print(str(len(li_failedimgs)) + ' failed images')
					print(li_failedimgs)

			print('sleeping...')
			time.sleep(8)

	df_failedimgs = pd.DataFrame()
	df_failedimgs['failed_hashes'] = li_failedimgs
	df_failedimgs.to_csv('mostused_img/' + querystring + '/failed_hashes_' + querystring + '.csv', mode='a', encoding='utf-8')

def getThumbImg(thumb_link, imghash, count_image, querystring, timesep):
	print('Trying to get thumbnail.')
	print(thumb_link)
	img_request = urllib.request.Request(thumb_link, headers=headers)
	
	try:
		img_response = urllib.request.urlopen(img_request)
	except urllib.error.HTTPError as httperror:
		print('HTTP error when requesting thread')
		print('Reason:', httperror.code)
		getThumbImg(thumb_link)
		pass
	else:
		print('Saving thumbnail')
		imagefile = io.BytesIO(img_response.read())
		image = Image.open(imagefile)
		imghash = imghash.replace('/','slash')
		if timesep:
			image.save('mostused_img/' + querystring + '/' + querystring + '_' + str(count_image) + '_' + imghash + '_thumb.' + str(image.format))
		else:
			image.save('mostused_img/' + querystring + '/' + querystring + '_' + str(count_image) + '_' + imghash + '_thumb.' + str(image.format))

getMostUsedImgs(querystring = 'mercury', hash_threshold = 10)