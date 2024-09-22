import os
import numpy as np
import pandas as pd
import re
import string
from collections import Counter
from gensim.utils import deaccent



def read_txt(path):

	files = open(path,encoding='utf-8-sig').read().split('\n\n')

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
	df_refs['first_author'] = [x for n,x in enumerate(first_author) if n not in ii]
	df_refs['authors'] = [x for n,x in enumerate(authors) if n not in ii]
	df_refs['article_title'] = [x for n,x in enumerate(article_title) if n not in ii]
	df_refs['abstracts'] = [x for n,x in enumerate(abstracts) if n not in ii]
	df_refs['source_title'] = [x for n,x in enumerate(source_title) if n not in ii]
	df_refs['publication_year'] = [x for n,x in enumerate(pub_year) if n not in ii]
	df_refs['ea'] = [x for n,x in enumerate(ea) if n not in ii]
	df_refs['journal_abbreviation'] = [x for n,x in enumerate(journal_abbr) if n not in ii]
	df_refs['journal_iso_abbreviation'] = [x for n,x in enumerate(journal_iso_abbr) if n not in ii]
	df_refs['references'] = [x for n,x in enumerate(refs) if n not in ii]
	df_refs['tot_cit'] = [x for n,x in enumerate(count_cit) if n not in ii]
	df_refs['doi'] = [x for n,x in enumerate(doi) if n not in ii]
	df_refs.drop_duplicates(['first_author','authors','article_title','publication_year'],inplace=True)
	df_refs.reset_index(inplace=True, drop=True)

	df_refs['text_id'] = ['d'+str(i) for i in range(df_refs.shape[0])]

	df = add_internal_citation(df_refs)
	df.to_excel('Dataset_input.xslx')
	return 


def add_top_journal(path,df):
	topj_list = [x.lower() for x in open(path).read().splitlines()]
	#j_list = [x.lower() for x in topj['JOURNAL TITLE']]
	df['TOPJ'] = ['N']*df.shape[0]
	df.loc[df[df['source_title'].isin(topj_list )].index,'TOPJ'] = 'Y'
	df.to_excel('Dataset_input.xslx')
	return

def add_internal_citation(df):
	df['references_internal_id'] = ['']*df.shape[0]
	df['references_residual'] = ['']*df.shape[0]

	for j,row in tqdm(df.iterrows()):
		cr = row['references']
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
						if df[(df['first_author']==au)&(df['publication_year']==int(y))&(df['journal_abbreviation']==journal)].shape[0]==1:
							temp.append(e)
							temp_idx.append(df[(df['first_author']==au)&(df['publication_year']==int(y))&(df['journal_abbreviation']==journal)]['text_id'].iloc[0])
							continue
						if df[(df['first_author']==au)&(df['publication_year']==int(y))&(df['journal_iso_abbreviation']==journal)].shape[0]==1:
							temp.append(e)
							temp_idx.append(df[(df['first_author']==au)&(df['publication_year']==int(y))&(df['journal_iso_abbreviation']==journal)]['text_id'].iloc[0]) 
							continue
						if df[(df['first_author']==au)&(df['publication_year']==int(y))&(df['source_title']==journal)].shape[0]==1:
							temp.append(e)
							temp_idx.append(df[(df['first_author']==au)&(df['publication_year']==int(y))&(df['source_title']==journal)]['text_id'].iloc[0]) 
							continue
					except:
						continue

		diff = set(all_temp).difference(temp)
		df.loc[j,'references_internal_id'] = ';'.join([str(x) for x in temp_idx])
		df.loc[j,'references_residual'] = ';'.join(list(diff))
	df['internal_cit']=[0]*df.shape[0]
	for j,row in df.iterrows():
		id_cit = row['references_internal_id']
		if id_cit!='':
			for e in set(id_cit.split(' ')):
				ii = df[df['text_id']==e].index[0]
				df.loc[ii,'internal_cit']+=1

	return df
	    





















