import os
import numpy as np
import pandas as pd
import re
import string
from collections import Counter
from tqdm import tqdm
from unidecode import unidecode
#from gensim.utils import deaccent

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
#stemmer = SnowballStemmer(language='english')
import spacy
#nlp = spacy.load('en_core_web_sm')#, disable=['tagger', 'ner','parser'])
#nlp.add_pipe(nlp.create_pipe('sentencizer'))
from spacy.lang.en.stop_words import STOP_WORDS



def cleaning(testo, other_stops=[]):
	sp=string.punctuation
	sp2=sp+'£'+'₹'+"‘"+"’"+ "”"+ "“" +"’"+"∗"+"’"+'©'

	StopWords = [x.lower() for x in set(stopwords.words('english')).union(STOP_WORDS)]
	StopWords2 = set()
	for e in StopWords:
		e2=e
		for p in sp2:
			e2 = e2.replace(p,' ')
		e3 = e2.split()
		for e4 in e3:
			StopWords2.add(e4)

	to_remove = [
	'originality/value'
	,'originality/valueto'
	,'originality/valuethe'
	,'originality/relevance'
	,'originality/valuethis'
	,'achievementsoriginality/valuethis'
	,'limitations/implications'
	,'limitations/implicationsthe'
	,'limitations/implicationsthis'
	,'Design/methodology/approach'
	,'design/methodology/approacha'
	,'design/methodology/approachto'
	,'design/methodology/approachthe'
	,'design/methodology/approachthis'
	,'actiondesign/methodology/approachin'
	,'actiondesign/methodology/approachour'
	,'actiondesign/methodology/approachthis'
	,'academiadesign/methodology/approachthe']

    
	text = re.sub(r"http\S+", " ", testo)
	text = re.sub(r'@\w+|＠\w+',' ',text)
	#text = deaccent(text)
	text = unidecode(text)
	text = re.sub(r'^Purpose','',text)
    
	for e in to_remove:
		rgx = r''+e
		text = re.sub(rgx,'',text,re.I)

	doc = nlp(text)
	list_sents = []
	Dict_word_stemmed = dict()
	for S in doc.sents:
		S2=str(S)
		S2 = re.sub(r'^Findings','',S2)
		text = re.sub(r'^Goal','',S2)
		text = re.sub(r'^Methodology','',S2)
		S2 = S2.lower()
		for punct in sp2:
			S2=S2.replace(punct,' ')
        
		new_sent = []
		for w in S2.split():
			Words=nlp(w)
			for token in Words:
				q = token.lemma_
				if '-PRON-'  in q:
					continue
				else:
					if q.isalpha() and len(q)>1 and len(w)>2 and q not in StopWords:
						stem = stemmer.stem(q)
						if stem not in other_stops:
							new_sent.append(stem)
							if Dict_word_stemmed.get(stem):
								Dict_word_stemmed[stem].add(w)
							else:
								Dict_word_stemmed.update({stem:set([w])})
                        
		list_sents.append(' '.join(new_sent) )
        
	corpus_final = ' . '.join(list_sents)

	return corpus_final, Dict_word_stemmed



def preprocess(df, col='abstracts'):
	stemmer = SnowballStemmer(language='english')
	nlp = spacy.load('en_core_web_sm')#, disable=['tagger', 'ner','parser'])
	nlp.add_pipe(nlp.create_pipe('sentencizer'))

	clean_text = []
	L_dict = []
	for t in tqdm(df['abstracts']):
		clean_t, dict_words = cleaning(t, other_stops)
		clean_text.append(clean_t)
		L_dict.append(dict_words)
	
	final_dict = dict()
	for d in L_dict: 
		for k,v in d.items():
			if final_dict.get(k):
				final_dict[k] = final_dict[k].union(v)
			else:
				final_dict.update({k:v})
	
	original_words = []
	for k,v in final_dict.items():
		original_words.append(' '.join(list(v)))
	
	df_stemming_role = pd.DataFrame()
	df_stemming_role['stemmed_word'] = final_dict.keys()
	df_stemming_role['original_words'] = original_words
	df_stemming_role.to_csv('stemming_role.csv',index=False)


	df['clean_text'] = clean_text
	#df.to_excel('Dataset_clean.xlsx',index=False)
	return df


