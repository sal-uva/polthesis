import substringFilter as subs
import similarities as sims
import getModel as getmod
import pickle
import sqlite3
import pandas as pd
import getTripleParsContent as threepars
import getNgrams as ngram

df = subs.substringFilter('kek', histogram=True, tocsv=False)