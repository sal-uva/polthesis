import sys
import os
import re
import pickle
import pandas as pd
import datetime
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

def createTokens(li_strings, stemming=False, lemmatizing=False, time=''):
	"""
	Saves tokens in pickle files.

	"""
	
	# Do some cleanup: only alphabetic characters, no stopwords
	# Create separate stemmed tokens, to which the full strings will be compared to:
	li_all_tokens = []
	len_comments = len(li_strings)
	
	print(len(li_strings))
	time_log = ''
	if time != '':
		time_log = ' for date slice ' + time
	print('Creating list of tokens' + time_log)
	
	for index, comment in enumerate(li_strings):
		
		# Create list of list for comments and tokens
		if isinstance(comment, str):
			li_tokens = []
			li_tokens = getFilteredText(comment, stemming=stemming, lemmatizing=lemmatizing)
			li_all_tokens.append(li_tokens)
		
		if index % 100 == 0:
			print('Tokenising finished for string ' + str(index) + '/' + str(len_comments))
	
	print('Saving tokens')
	if time == '':
		time = str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
	if not os.path.exists('output/'):
		os.makedirs('output/')
	pickle.dump(li_all_tokens, open('output/tokens-' + time + '.p', 'wb'))

def getFilteredText(string, stemming=False, lemmatizing=False):
	
	#first, remove urls
	if 'http' in string:
		string = re.sub(r'https?:\/\/.*[\r\n]*', ' ', string)
	if 'www.' in string:
		string = re.sub(r'www.*[\r\n]*', ' ', string)

	# get a list of words
	tokens = re.findall("[a-zA-Z\-\)\(]{3,50}", string)

	stemmer = SnowballStemmer("english")
	lemmatizer = WordNetLemmatizer()

	#list with tokens further processed
	li_filtered_tokens = []
	
	# filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
	for token in tokens:
		token = token.lower()
		
		# only alphabetic characters, keep '(' and ')' symbols for echo brackets, only tokens with three or more characters
		if re.match('[a-zA-Z\-\)\(]{3,50}', token) is not None:
			# no stopwords
			if token not in stopwords.words('english'):
				token = token.lower()
				# shorten word if it's longer than 20 characters (e.g. 'reeeeeeeeeeeeeeeeeeeeeeeee')
				if len(token) >= 20:
					token = token[:20]
				
				# stem if indicated it should be stemmed
				if stemming:
					token_stemmed = stemmer.stem(token)
					li_filtered_tokens.append(token_stemmed)

					# update lookup dict with token and stemmed token
					# lookup dict is dict of stemmed words as keys and lists as full tokens
					if token_stemmed in di_stems:
						if token not in di_stems[token_stemmed]:
							di_stems[token_stemmed].append(token)
					else:
						di_stems[token_stemmed] = []
						di_stems[token_stemmed].append(token)
				
				#if lemmatizing is used instead
				elif lemmatizing:
					
					token = lemmatizer.lemmatize(token)
					
					# lemmatize 'hahahaha' - needed for 'kek' w2v similarities
					if token[:5] == 'hahah':
						token = 'haha'
					elif token[:5] == 'ahaha':
						token = 'ahah'

					li_filtered_tokens.append(token)
				
				else:
					li_filtered_tokens.append(token)

	return li_filtered_tokens

# Show manual if needed
if len(sys.argv) < 2:
	print()
	print("Tokenizes text from a csv column.")
	print("Saves the results in binary files (using the library pickle).")
	print("Can stem and/or lemmatize.")
	print()
	print("Usage: python3 createTokens.py [--source] [--text] [--stem] [--lemmatize] [--timespan]")
	print()
	print("--source: the relative path to a csv file from 4CAT (e.g. 'data/datasheet.csv').")
	print("--text: default 'body' - the csv column with the text.")
	print("--stem: default false - whether to stem the text.")
	print("--lemma: default true - whether to lemmatize the text.")
	print("--timespan: if provided, create output per month/day. Use 'month' or 'day'.")
	print("--timecol: default 'timespan' - the csv column in which the time values are stored. Should start with format dd-mm-yyyy.")
	print()
	print("Example: python getTokens.py --data/4cat-datasheet.csv --text=body --stem=False --lemma=True --timespan=month --timecol=timestamp")
	print()
	sys.exit(1)

else:
	li_args = []
	source = ''
	text_column = 'body'
	time_column = 'timestamp'
	stem = False
	lemma = True
	timespan = False

	# Interpret command line arguments
	for arg in sys.argv:
		if arg[0:9] == "--source=":
			source = arg[9:len(arg)]
			li_args.append(source)
		elif arg[0:7] == "--text=":
			text_column = arg[7:len(arg)]
			li_args.append(text_column)
		elif arg[0:7] == "--stem=":
			stem_input = arg[7:len(arg)]
			if stem_input == 'true':
				stem = True
			li_args.append(stem)
		elif arg[0:8] == "--lemma=":
			lemma_input = arg[8:len(arg)]
			if lemma_input == 'false':
				lemma = False
			li_args.append(lemma)
		elif arg[0:11] == "--timespan=":
			timespan = arg[11:len(arg)]
			li_args.append(timespan)
		elif arg[0:10] == "--timecol=":
			time_column = arg[10:len(arg)]
			li_args.append(time_column)
	print(li_args)
	if source == '' or not os.path.isfile(source):
		print("Please provide a valid input file like this: --source=data/datasheet.csv")
		sys.exit(1)

	else:
		df = pd.read_csv(source)
		df = df[:300]

		if timespan == False:
			li_input = df[text_column].tolist()
			createTokens(li_input, stemming=stem, lemmatizing=lemma)
		else:
			print('Tokenizing per ' + timespan)
			# Get the dates to make files for
			li_all_dates = df[time_column].tolist()
			if timespan == 'days':
				li_dates = [date[0:10] for date in li_all_dates]
			elif timespan == 'months':
				li_dates = [date[3:10] for date in li_all_dates]
			else:
				print('Please provide a valid date format (\'days\' or \'months\')')
				sys.exit(1)
			df['dates_to_check'] = li_dates
			li_dates = set(li_dates)

			for date_slice in li_dates:
				df_date = df[df['dates_to_check'] == date_slice]
				li_input = df_date[text_column].tolist()
				createTokens(li_input, stemming=stem, lemmatizing=lemma, time=date_slice)