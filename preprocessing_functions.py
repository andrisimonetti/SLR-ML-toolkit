import os
from unidecode import unidecode
import re
import string
import numpy as np
import pandas as pd
from collections import Counter
from gensim.utils import deaccent
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import spacy
from spacy.lang.en.stop_words import STOP_WORDS




def read_zootero_csv(filename):
	df = pd.read_csv(filename)
	selected_columns = ['Author', 'Title', 'Abstract Note','Publication Title', 'Publication Year', 'Journal Abbreviation','References', 'DOI']
	df = df.loc[:,selected_columns]
	df.rename(columns={'Author':'First author','Title': 'Article title', 'Abstract Note': 'Abstracts','Publication Title': 'Source title', 'Publication Year':'Publication year',
		'Journal Abbreviation':'Journal abbreviation','DOI':'doi'}, inplace=True)

def read_scopus_csv(filename):
	df = pd.read_excel(filename)
	selected_columns = ['Authors', 'Title', 'Abstract','Source title', 'Year', 'Abbreviated source title','References', 'Cited by','DOI']
	selected_columns = list(set(df.columns).intersection(selected_columns))
	df = df.loc[:,selected_columns]
	df.rename(columns={'Title': 'Article title', 'Abstract': 'Abstracts','Source Title': 'Source title', 'Year':'Publication year',
		'Cited by':'Total number of citations','Abbreviated source title':'Journal abbreviation','DOI':'doi'}, inplace=True)
	df['First author'] = [x.split(',')[0] for x in df['Authors']]

	df.drop_duplicates(['First author','Authors','Article title','Publication year'],inplace=True)
	df.reset_index(inplace=True, drop=True)
	df['text_id'] = ['d'+str(i) for i in range(df.shape[0])]
	#df = add_internal_citation(df)
	writer = pd.ExcelWriter('Dataset_input.xlsx',
                        engine='xlsxwriter',
                        engine_kwargs={'options': {'strings_to_urls': False}})
	df.to_excel(writer)
	writer.close()
	#df.to_excel('Dataset_input.xlsx',index=False)
	return

def read_wos_txt(filename):

	files = open(filename,encoding='utf-8-sig').read().split('\n\n')

	n_files = len(files)
	authors = ['']*n_files
	first_author = ['']*n_files
	source_title = ['']*n_files
	article_title = ['']*n_files
	pub_year = ['']*n_files
	refs = ['']*n_files
	count_cit = ['']*n_files
	abstracts = ['']*n_files
	journal_abbr = ['']*n_files
	journal_iso_abbr = ['']*n_files
	doi = ['']*n_files
	ea = ['']*n_files


	for j,e in enumerate(files):
	    e = e.replace('<i>','')
	    e = e.replace('</i>','')
	    cols = e.split('\n')
	    start=0
	    #end1=0
	    end2 = 0
	    for n,c in enumerate(cols):
	        if c[:2]=='AF':
	            authors[j]= deaccent(c.replace('AF ','').lower())
	            for ii in range(n+1,n+4):
	                if cols[ii][:2]=='  ':
	                    authors[j]=authors[j]+';'+deaccent(cols[ii].replace('   ','').lower())
	                else:
	                    break
	            
	        if c[:2]=='AU':
	            first_author[j]= deaccent(c.replace('AU ','').lower())
	        if c[:2]=='TI':
	            article_title[j]= deaccent(c.replace('TI ','').lower())
	            
	            for ii in range(n+1,n+4):
	                if cols[ii][:2]=='  ':
	                    article_title[j]=article_title[j]+' '+cols[ii].replace('   ','').lower()
	                else:
	                    break
	                    
	        if c[:2]=='SO':
	            source_title[j]= deaccent(c.replace('SO ','').lower())
	            for ii in range(n+1,n+4):
	                if cols[ii][:2]=='  ':
	                    source_title[j]=source_title[j]+' '+cols[ii].replace('   ','').lower()
	                else:
	                    break
	                    
	        if c[:2]=='CR':
	            refs[j] = c.replace('CR ','')
	            for ii in range(n+1,n+500):
	                if cols[ii][:2]=='  ':
	                    refs[j]=refs[j]+';'+cols[ii].replace('   ','')
	                else:
	                    break
	                    
	        if c[:2]=='AB':# and cols[n+1][:2]=='C1':
	            abstracts[j] = c.replace('AB ','')
	            for ii in range(n+1,len(cols)):
	                if cols[ii][:2]!='C1':
	                    abstracts[j]=abstracts[j]+' '+cols[ii].replace('   ','')
	                else:
	                    break
	            
	        if c[:2]=='PY':
	            pub_year[j]=int(c.replace('PY ',''))
	        if c[:2]=='J9':
	            journal_abbr[j]=c.replace('J9 ','')
	        if c[:2]=='JI':
	            journal_iso_abbr[j]=c.replace('JI ','')
	        if c[:2]=='TC':
	            count_cit[j]=int(c.replace('TC ',''))
	        if c[:2]=='DI':
	            doi[j]=c.replace('DI ','')
	        if c[:2]=='EA':
	            ea[j]=c.replace('EA ','')


	ii = [n for n,x in enumerate(abstracts) if x=='']
	df_refs = pd.DataFrame()
	df_refs['First author'] = [x for n,x in enumerate(first_author) if n not in ii]
	df_refs['Authors'] = [x for n,x in enumerate(authors) if n not in ii]
	df_refs['Article title'] = [x for n,x in enumerate(article_title) if n not in ii]
	df_refs['Abstracts'] = [x for n,x in enumerate(abstracts) if n not in ii]
	df_refs['Source title'] = [x for n,x in enumerate(source_title) if n not in ii]
	df_refs['Publication year'] = [x for n,x in enumerate(pub_year) if n not in ii]
	#df_refs['ea'] = [x for n,x in enumerate(ea) if n not in ii]
	df_refs['Journal abbreviation'] = [x for n,x in enumerate(journal_abbr) if n not in ii]
	df_refs['Journal iso abbreviation'] = [x for n,x in enumerate(journal_iso_abbr) if n not in ii]
	df_refs['References'] = [x for n,x in enumerate(refs) if n not in ii]
	df_refs['Total number of citations'] = [x for n,x in enumerate(count_cit) if n not in ii]
	df_refs['doi'] = [x for n,x in enumerate(doi) if n not in ii]
	df_refs.drop_duplicates(['First author','Authors','Article title','Publication year'],inplace=True)
	df_refs.reset_index(inplace=True, drop=True)

	df_refs['text_id'] = ['d'+str(i) for i in range(df_refs.shape[0])]

	df = add_internal_citation_wos(df_refs)
	df_refs.to_excel('Dataset_input.xlsx',index=False)
	return 


def add_top_journal(filename, df_file):
	topj_list = [x.lower() for x in open(filename).read().splitlines()]
	#j_list = [x.lower() for x in topj['JOURNAL TITLE']]
	df = pd.read_excel(df_file)
	df['TOPJ'] = ['N']*df.shape[0]
	df.loc[df[df['Source title'].isin(topj_list )].index,'TOPJ'] = 'Y'
	df.to_excel('Dataset_input.xlsx')
	return

def add_internal_citation_wos(df):
	df['References internal id'] = ['']*df.shape[0]
	#df['References residual'] = ['']*df.shape[0]

	for j,row in tqdm(df.iterrows()):
		cr = row['References']
		all_temp = []
		temp = []
		temp_idx = []
		if cr!='':
			for e in cr.split(';'):
				all_temp.append(e)
				doi = re.search(r'DOI .*',e)
				if doi and df[df['doi']==doi.group(0).replace('DOI ','')].shape[0]==1:
					temp.append(e)
					temp_idx.append(df[df['doi']==doi.group(0).replace('DOI ','')]['text_id'].iloc[0])
					continue
				else:
					try:
						au = deaccent(e.split(',')[0].split()[0].replace(',',''))
						y = deaccent(e.split(',')[1].split()[0])
						journal = e.split(',')[2].lower()
						if df[(df['First author']==au)&(df['Publication year']==int(y))&(df['Source title']==journal)].shape[0]==1:
							temp.append(e)
							temp_idx.append(df[(df['First author']==au)&(df['Publication year']==int(y))&(df['Source title']==journal)]['text_id'].iloc[0]) 
							continue
						if df[(df['First author']==au)&(df['Publication year']==int(y))&(df['Journal abbreviation']==journal)].shape[0]==1:
							temp.append(e)
							temp_idx.append(df[(df['First author']==au)&(df['Publication year']==int(y))&(df['Journal abbreviation']==journal)]['text_id'].iloc[0])
							continue
						if df[(df['First author']==au)&(df['Publication year']==int(y))&(df['Journal iso abbreviation']==journal)].shape[0]==1:
							temp.append(e)
							temp_idx.append(df[(df['First author']==au)&(df['Publication year']==int(y))&(df['Journal iso abbreviation']==journal)]['text_id'].iloc[0]) 
							continue
					except:
						continue

		diff = set(all_temp).difference(temp)
		df.loc[j,'References internal id'] = ' '.join([str(x) for x in temp_idx])
		#df.loc[j,'References residual'] = ';'.join(list(diff))
	df['Number of internal citations']=[0]*df.shape[0]
	for j,row in df.iterrows():
		id_cit = row['References internal id']
		if id_cit!='':
			for e in set(id_cit.split(' ')):
				ii = df[df['text_id']==e].index[0]
				df.loc[ii,'Number of internal citations']+=1

	return df

def cleaning(testo, other_stops=[]):
	stemmer = SnowballStemmer(language='english')
	nlp = spacy.load('en_core_web_sm')#, disable=['tagger', 'ner','parser'])
	nlp.create_pipe('sentencizer')
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



def preprocess(filename, col='Abstracts'):
	df = pd.read_excel(filename)

	clean_text = []
	L_dict = []
	for t in tqdm(df['Abstracts']):
		clean_t, dict_words = cleaning(t, other_stops=[])
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
	df_stemming_role['word'] = final_dict.keys()
	df_stemming_role['original words'] = original_words
	df_stemming_role.to_csv('stemming_role.csv',index=False)


	df['clean_text'] = clean_text
	#df.to_excel('Dataset_clean.xlsx',index=False)
	df.to_excel('Dataset_input.xlsx',index=False)
	return
	    





















