import sys
import os
import re

def createLongString(inputdf, querystring, currenttime):
	directory = 'substring_mentions/longstring_' + querystring.replace(' ', '-')

	if not os.path.exists(directory):
		os.makedirs(directory)

	# Write a txt file with all the strings
	txtfile_full = open(directory + '/longstring_' + querystring + '_full.txt', 'a', encoding='utf-8')
	
	# Write time separated txt files
	txtfile_sep = open(directory + '/longstring_' + querystring + '_' + currenttime + '.txt', 'w', encoding='utf-8')
	str_keyword = ''
	li_str = []

	for item in inputdf['comments']:
		item = item.lower()
		#regex = re.compile("[^a-zA-Z \.\,\-\n]")		# old regex, excludes numbers
		regex = re.compile('[^a-zA-Z\)\(\.\,\-\n ]')	# includes brackets
		item = regex.sub('', item)
		txtfile_sep.write('%s' % item)
		txtfile_full.write('%s' % item)
		str_keyword = str_keyword + item
		li_str.append(item)

	return str_keyword, li_str

# show manual if needed
if len(sys.argv) < 2:
	print()
	print("Creates a txt file of a long sequence of a column in a csv.")
	print("Useful to make Word Tree visualisations.")
	print()
	print("Usage: python3 createLongString.py [--source] [--output] [--timespan] [--timecolumn]")
	print()
	print("--source: the relative path to a csv file from 4CAT (e.g. 'data/datasheet.csv').")
	print("--string: the string that was filtered on in the csv. Used as plot title and output file.")
	print("--timespan: default 'days' - the separation of the histogram bars. Can be 'days' or 'months'.")
	print("--timecolumn: default 'timespan' - the csv column in which the time values are stored. Should start with format dd-mm-yyyy.")
	print()
	print("Example: python createHistogram.py --input=data/4cat-datasheet.csv --string=obama --timespan=months --timecolumn=months")
	print()
	sys.exit(1)