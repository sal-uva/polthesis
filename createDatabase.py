import pandas as pd
import sqlite3
from sqlalchemy import create_engine

def createDatabase():
	pol_database = create_engine('sqlite:///poldatabase.db')

	# quit()

	file = 'poldatabase.csv'


	chunksize = 100000
	i = 0
	j = 1
	#columns = ['num', 'subnum', 'thread_num', 'op', 'timestamp', 'timestamp_expired', 'preview_orig', 'preview_w', 'preview_h', 'media_filename', 'media_w', 'media_h', 'media_size', 'media_hash', 'media_orig', 'spoiler', 'deleted', 'capcode', 'email', 'name', 'trip', 'title', 'comment', 'sticky', 'locked', 'poster_hash', 'poster_country', 'exif']

	for df in pd.read_csv(file, escapechar='\\', encoding='utf-8', engine='python', chunksize=chunksize, iterator=True):
	      #df = df.rename(columns={c: c.replace(' ', '') for c in df.columns}) 
	      df.columns = columns
	      
	      df.index += j
	      i+=1
	      df.to_sql('postcontent', pol_database, index=False, if_exists='append')
	      j = df.index[-1] + 1
	      print(str(i))

conn = sqlite3.connect("../4plebs_pol_18_03_2018.db")
df = pd.read_sql_query("SELECT * FROM poldatabase_18_03_2018 LIMIT 2000", conn)
pol_database = create_engine('sqlite:///pol_database.db')
df.to_sql('poltable', pol_database, index=False, if_exists='append')