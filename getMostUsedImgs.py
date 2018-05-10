import sqlite3
import pandas as pd
import io
import os
import urllib.request, json
import time
import substringFilter
from PIL import Image

# HEADERS: num, subnum, thread_num, op, timestamp, timestamp_expired, preview_orig,
# preview_w, preview_h, media_filename, media_w, media_h, media_size,
# media_hash, media_orig, spoiler, deleted, capcode, email, name, trip,
# title, comment, sticky, locked, poster_hash, poster_country, exif

# test db: 4plebs_pol_test_database
# test table: poldatabase
# full db: 4plebs_pol_18_03_2018
# full table: poldatabase_18_03_2018

user_agent = 'Mozilla/5.0 (Windows NT 6.1; Win64; x64)'
headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
   'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
   'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
   'Accept-Encoding': 'none',
   'Accept-Language': 'en-US,en;q=0.8',
   'Connection': 'keep-alive'}
<<<<<<< HEAD

def getMostUsedImgs(querystring=None, stringintitle=False, downloadimg_thres=1000):
	querystring = querystring.lower()

	print('Connecting to database')
	conn = sqlite3.connect("../4plebs_pol_18_03_2018.db")
=======
>>>>>>> 5570f07e0fb03ec6b32d591d0b79901021db1150

def getMostUsedImgs(querystring=None, separate_month=False, stringintitle=False, hash_threshold=0, loadcsv=''):
	querystring = querystring.lower()
	li_failedimgs = []

<<<<<<< HEAD
=======
	#if a csv is loaded, load csv, else check the database
>>>>>>> 5570f07e0fb03ec6b32d591d0b79901021db1150
	if loadcsv != '':
		print('Reading csv file')
		df = pd.read_csv(loadcsv, encoding='utf-8')
	else:
		print('Connecting to database')
		conn = sqlite3.connect("../4plebs_pol_test_database.db")

		print('Beginning SQL query for "' + querystring + '"')
		if querystring == 'all':
			querystring = querystring + '-' + str(datetime.strftime(datetime.fromtimestamp(mintime), "%m-%Y"))
			df = pd.read_sql_query("SELECT timestamp, comment, media_hash FROM poldatabase WHERE timestamp > ? AND timestamp < ?;", conn, params=[mintime, maxtime])
		#look for string in subject
		elif stringintitle == False:
			df = pd.read_sql_query("SELECT timestamp, comment, media_hash FROM poldatabase WHERE lower(comment) LIKE ?;", conn, params=['%' + querystring + '%'])
		#look for sting in comment body (default)
		else:
			df = pd.read_sql_query("SELECT timestamp, title, media_hash FROM poldatabase WHERE lower(title) LIKE ?;", conn, params=['%' + querystring + '%'])
	
	if separate_month == True:
		for month in li_monthstamps:
			df_month = 
			getImgs(df=df_month, querystring=querystring, hash_threshold=hash_threshold, timesep=True)
	else:
		getImgs(df=df, querystring=querystring, hash_threshold=hash_threshold, timesep=False)

def getImgs(df, querystring, hash_threshold, timesep=False):
	# create df with grouped and descending hashes
	df['occurrances'] = [1] * len(df)
	df = df.groupby('media_hash').agg({'occurrances': len})
	df = df.sort_values(by=['occurrances'], ascending=False)
	#df.rename(columns={'media_hash': 'hash', 'occ': 'occurances'})
	df['hash'] = df.index.values
	df.reset_index(level=0, inplace=True)
	print(df.head())
	# make folder
	if os.path.exists('mostused_img/'+ querystring +'/') == False:
		os.makedirs('mostused_img/'+ querystring +'/')
	print(hash_threshold)
	for index, count_image in enumerate(df['occurrances']):
		imghash = df['hash'][index]
		if count_image >= hash_threshold:
			print('above threshold')
			url = 'http://archive.4plebs.org/_/api/chan/search/?boards=pol&image=' + imghash
			print(url)
			request = urllib.request.Request(url, headers=headers)
			#request.add_header()
			#json with post data
			try:								#check if the thread is still active on the site
				response = urllib.request.urlopen(request)
				# imageurl = "http://i.4cdn.org/pol/" + str(imagename)

			except urllib.error.HTTPError as httperror:                       #some threads get deleted and return a 404
				print('HTTP error when requesting thread')
				print('Reason:', httperror.code)
				pass
			else:
				data = response.read().decode('utf-8', 'ignore')
				postdata = json.loads(data)

				if 'error' not in postdata:
					#loop through posts with image hash until image is downloaded (some are invalid)
					print(len(postdata['0']['posts']))

					#for index, post in enumerate(postdata['0']['posts']):
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
							if '.webm' not in imagefile:
								imagefile = io.BytesIO(img_response.read())
								image = Image.open(imagefile)
								# print('imagesize: ' + str(image.size))
								imagesize = image.size
								#if imagesize[0] > 800 or imagesize[1] > 800:
									# print('Resizing...')
									#image.thumbnail(size)
								imghash = imghash.replace('/','slash')
								image.save('mostused_img/' + querystring + '/' + querystring + '_' + str(count_image) + '_' + imghash + '.' + str(image.format))
					else:
						print('no media found')
					print('sleeping...')
					time.sleep(12)
				else:
<<<<<<< HEAD

=======
>>>>>>> 5570f07e0fb03ec6b32d591d0b79901021db1150
					li_failedimgs.append(imghash)
					print('invalid image')
					print(str(len(li_failedimgs)) + ' failed images')
					print(li_failedimgs)

			print('sleeping...')
			time.sleep(12)
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
		# print('imagesize: ' + str(image.size))
		imagesize = image.size
		#if imagesize[0] > 800 or imagesize[1] > 800:
			# print('Resizing...')
			#image.thumbnail(size)
		imghash = imghash.replace('/','slash')
		if timesep:
			image.save('mostused_img/' + querystring + '/' + querystring + '_' + str(count_image) + '_' + imghash + '_thumb.' + str(image.format))
		else:
			image.save('mostused_img/' + querystring + '/' + querystring + '_' + str(count_image) + '_' + imghash + '_thumb.' + str(image.format))
							
<<<<<<< HEAD
getMostUsedImgs(querystring='skyrim', separate_months=True, hash_threshold=0, loadcsv='mentions_comment_skyrim__.csv')
					print(postdata)
					print('invalid, trying thumb')
					time.sleep(13)
					user_agent = 'Mozilla/5.0 (Windows NT 6.1; Win64; x64)'
					headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
				       'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
				       'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
				       'Accept-Encoding': 'none',
				       'Accept-Language': 'en-US,en;q=0.8',
				       'Connection': 'keep-alive'}
					url = 'http://archive.4plebs.org/_/api/chan/search/?boards=pol&image=' + imghash
					print(url)
					request = urllib.request.Request(url, headers=headers)
					#request.add_header()
					#json with post data
					try:								#check if the thread is still active on the site
						response = urllib.request.urlopen(request)
						# imageurl = "http://i.4cdn.org/pol/" + str(imagename)

					except urllib.error.HTTPError as httperror:                       #some threads get deleted and return a 404
						print('HTTP error when requesting thread')
						print('Reason:', httperror.code)
						pass
					else:
						#if 
						data = response.read().decode('utf-8', 'ignore')
						postdata = json.loads(data)

						if 'error' not in postdata:
							postdata = postdata['0']['posts'][0]['media']
							#print(postdata)
							if 'thumb_link' in postdata:
								print('image found')
								img_link = postdata['thumb_link']
								print(img_link)
								img_request = urllib.request.Request(img_link, headers=headers)
								try:
									img_response = urllib.request.urlopen(img_request)
								except urllib.error.HTTPError as httperror:
									print('HTTP error when requesting thread')
									print('Reason:', httperror.code)
									pass
								else:
									imagefile = io.BytesIO(img_response.read())
									image = Image.open(imagefile)
									# print('imagesize: ' + str(image.size))
									imagesize = image.size
									#if imagesize[0] > 800 or imagesize[1] > 800:
										# print('Resizing...')
										#image.thumbnail(size)
									imghash = imghash.replace('/','slash')
									image.save('mostused_img/' + querystring + '/' + querystring + '_' + str(count_image) + '_' + imghash + '.' + str(image.format))
							else:
								print('no media found')
						else:
							print(postdata)
							print('invalid')
					
			time.sleep(13)

getMostUsedImgs(querystring='trump', downloadimg_thres=25)
=======
getMostUsedImgs(querystring='skyrim', separate_months=True, hash_threshold=0, loadcsv='mentions_comment_skyrim__.csv')
>>>>>>> 5570f07e0fb03ec6b32d591d0b79901021db1150
