import pandas as pd
import urllib.request, json

threadnos = [177069196,177105345,177134316,177152400,177182368,177225375,177257660,177280259,177331566,177368305,177404301,177436598,177436662]
user_agent = 'Mozilla/5.0 (Windows NT 6.1; Win64; x64)'
headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
   'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
   'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
   'Accept-Encoding': 'none',
   'Accept-Language': 'en-US,en;q=0.8',
   'Connection': 'keep-alive'}

df = pd.DataFrame()
for threadno in threadnos:
	url = 'http://archive.4plebs.org/_/api/chan/thread/?board=pol&num=' + str(threadno)
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
		print(postdata)

		#put OP in df
		df1 = pd.DataFrame.from_dict(postdata[str(threadno)]['op'], orient='index')
		df1 = df1.transpose()

		#put posts in df
		df_post = pd.DataFrame.from_dict(postdata[str(threadno)]['posts'], orient='index')
		#df_post = df_post.transpose()
		frames = [df1, df_post]
		df1 = pd.concat(frames)
		print(df1)
		frames = [df, df1]
		df = pd.concat(frames)
	
	df.to_csv('syria_generals.csv', encoding='utf-8')
		
