import sqlite3
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import datetime
import operator
import re
import os.path

# HEADERS: num, subnum, thread_num, op, timestamp, timestamp_expired, preview_orig,
# preview_w, preview_h, media_filename, media_w, media_h, media_size,
# media_hash, media_orig, spoiler, deleted, capcode, email, name, trip,
# title, comment, sticky, locked, poster_hash, poster_country, exif

# test db: 4plebs_pol_test_database
# test table: poldatabase
# full db: 4plebs_pol_18_03_2018
# full table: poldatabase_18_03_2018

def getTripleParsContent(inmonth = '', fb = False):
	print('Getting content in triple parentheses')
	#if there's already a csv with ALL mentions of triple parentheses
	if fb:
		postcolumn = 'message'
	else:
		postcolumn = 'post'
	if os.path.isfile('substring_mentions/triplepars.csv'):
		print('Reading triplepars.csv')
		df = pd.read_csv('substring_mentions/triplepars.csv', encoding='utf-8')
	#if the file's not there yet, fetch it from the database
	else:
		if fb:
			print('Running SQL query on the FB database')
			conn = sqlite3.connect("../fb_scrapelist_alldata.db")
			SQLquery = "SELECT created_time, page_id, page, message FROM fbpagedata WHERE (lower(message) LIKE '%(((%' AND lower(message) LIKE '%)))%');"
			df = pd.read_sql_query(SQLquery, conn)
			li_anglopages_ids = ['164305410295882','423006854503882','427302110732180','95475020353','15139936162','300455573433044','58158120046','80256732576','381971441938916','193266897438','129617873765147','313210635511804','1499825823571529','1477765039129741','150012808367284','145917588805718','24330467048','71523830069','811793512220923','209101162445115','120984578858','1078512562181459','1501449833505858','1533411596935910','350693121790195','819086604786287','184795298567879','32125256650','44951363957','415483818608473','195065046451','18017481462','214160718649383','161358883953210','153635028036616','376566465793805','762398587169729','359762154043841','384177565016061','388162771573834','189787347717980','132296387281990','113975382017986','101095043361','145684232179862','821897474624093','853653021326097','105891336235920','252173821949581','955319097860174','23159793458','162167430574227','503462116462653','281347662044844','273793279377333','781738638551730','606294432781528','967502819947271','1911491975742856','163100830919353','1061921623935380','659685410802725','118907334848560','1475431146051651','97616068669','316414711900570','750497664972072','538906699602145','564267373648165','322751537914061','149677321901297','189451037766669','1542124472708961','177430395633158','1440007272962796','355621771140731','513059668856085','210813579250854','926680257408193','125423387486286','130034640359470','146355585379031','473588786071833','1553098468242274','115866888428103','254086628012065','242889362473289','796304153798944','1483886811826197','569270129851848']
			li_anglopages_names = ['Daily Mail','Milo Yiannopoulos','Mike Cernovich','Breitbart','Steven Crowder','Britain First','American Center for Law and Justice','InfoWars','Tommy Robinson','Pamela Geller','Daily Express','Never Again Canada','Reclaim Australia Rally','Guardians Of Australia','The Australian Tea Party','Dr Jordan B Peterson','Americans for Prosperity','British National Party','The Rebel','UK Independence Party (UKIP)','Lew Rockwell','EDL - English Defence League','Birmingham Infidels UK','Pegida Canada','Reclaim Australia Rally - Western Australia','Australian Liberty Alliance','Lauren Southern','The American Conservative','American Enterprise Institute','Stop the islamization of the USA  Pegida USA','Ukrainian Canadian Congress - UCC National','Acton Institute','Hellenic American Leadership Council','Rise Up Australia Party','North East English Defence League','Liberty GB','Anti Islam - Australia','Black Pigeon Speaks','English Defence League Sikh Division - EDL','Soldiers Of Odin USA','EDL English Defence League Royal Berkshire Division','Generation Identity United Kingdom and the Republic of Ireland','Rand Paul Revolution','Americans United for Life','AFDI American Freedom Defense Initiative','New British Union of Fascists','National Front','Irish Patriot Movement','Generation Identity Alba/Scotland','Sworn American Patriots with Joe Sworn','Pat Condell','English Defence League. EDL. Lancaster Division.','Pegida promotional page  Scotland','Politically Incorrect Australia','Acts 17 Apologetics','Identity Ireland','EDL Southern angels of the English defence league','Pegida Canada Alberta','English Defence League Essex Division','Soldiers of Odin Canada Support','Sharia Watch UK Ltd.','Pegida Canada BC','BNPtv','EDL/ fleetwood division','Jordan Crowder','Anti Islam - Australia','EDL Kent Angels','Pegida Ireland','Millennial Woes','Pegida northern Ireland','American Brotherhood MC','Weymouth English Defence League EDL','Stand Up For Australia - Bunbury Western Australia','English Defence League (EDL Exeter Division) Members and Supporters','UKIP Local','Grimsby Division EDL','Soldiers of Odin UK','LGBT Supporters of Pegida UK','Pegida NZ','AEI Program on American Citizenship','EDL Wiltshire Division','EDL Dorset Division','Hrvatska Stranka Prava Croatian Party of Rights Hsp Inc. Australia','EDL / English Defence League / Blackpool Division','UKIP','EDL - Colchester  Division of the English Defence League','EDL Salisbury Division English Defence League','Reclaim Australia Rally - Bunbury','Syria Solidarity Ireland','American Brotherhood']
			#df = df[(df['page'].isin(li_anglopages_names)) | (df['page_id'].isin(li_anglopages_ids))]
			df.to_csv('substring_mentions/triplepars.csv', encoding='utf-8')
		else:
			print('Running SQL query on the 4plebs database')
			conn = sqlite3.connect("../4plebs_pol_18_03_2018.db")
			SQLquery = "SELECT comment, date_full, date_month, timestamp FROM pol_content WHERE (lower(comment) LIKE '%(((%' AND lower(comment) LIKE '%)))%');"
			df = pd.read_sql_query(SQLquery, conn)
			df.to_csv('substring_mentions/triplepars-4chan.csv', encoding='utf-8')
			quit()
	
	if fb:
		li_months = []
		for fulldate in df['created_time']:
			li_months.append(fulldate[:7])
		df['date_month'] = li_months

	df = df[df['date_month'] == inmonth]
	li_parscontents = []

	print('Extracting triple pars contents from DataFrame')
	for string in df[postcolumn]:
		li_matches = re.findall(r'[\(]{3,}(.*?)[\)]{3,}', string)
		for match in li_matches:
			if match != '':
				match = match.lower().strip()
				li_parscontents.append(match)

	print(li_parscontents[:9])
	print('Writing txt file')

	di_all_parscontents = {}
	for parscontent in li_parscontents:
		if parscontent not in di_all_parscontents:
			di_all_parscontents[parscontent] = 1
		else:
			di_all_parscontents[parscontent] += 1
	result = sorted(di_all_parscontents.items(), key=operator.itemgetter(1), reverse=True)
	print(result)

	write_handle = open('substring_mentions/triplepars_' + inmonth + '.txt',"w", encoding='utf-8')
	write_handle.write(str(result))
	write_handle.close()
	df = pd.DataFrame.from_dict(result)
	df.to_csv('substring_mentions/triplepars_' + inmonth + '.csv', encoding='utf-8')
	return df