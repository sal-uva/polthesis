import re
import pickle
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

def getTokens(li_strings, stemming=False, lemmatizing=False):
	"""
	Self-made tokenizer for chan text

	"""

	if stemming:
		global di_stems
		di_stems = pickle.load(open('di_stems.p', 'rb'))

	print('imported')
	
	# Do some cleanup: only alphabetic characters, no stopwords
	# Create separate stemmed tokens, to which the full strings will be compared to:
	li_comments_stemmed = []
	len_comments = len(li_strings)
	
	print(len(li_strings))
	print('Creating list of tokens per monthly document')
	
	for index, comment in enumerate(li_strings):
		
		# Create list of list for comments and tokens
		if isinstance(comment, str):
			li_comment_stemmed = []
			li_comment_stemmed = getFilteredText(comment, stemming=stemming, lemmatizing=lemmatizing)
			li_comments_stemmed.append(li_comment_stemmed)
		
		if index % 10000 == 0:
			print('Tokenising finished for string ' + str(index) + '/' + str(len_comments))
	
	print(len(li_comments_stemmed))

	if stemming:
		pickle.dump(di_stems, open('di_stems.p', 'wb'))
		df_stems = pd.DataFrame.from_dict(di_stems, orient='index')
		df_stems.to_csv('di_stems_dataframe.csv', encoding='utf-8')

	return li_comments_stemmed

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