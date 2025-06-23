import os
import re
import string
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from scipy.stats import hypergeom
import networkx as nx
import matplotlib.pyplot as plt
from unidecode import unidecode
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import spacy
from spacy.lang.en.stop_words import STOP_WORDS


def correlation(ni,nj,nij,n):
	rho=0
	if ni!=n and nj!=n:
		rho = (nij-(ni*nj)/n)/np.sqrt(ni*(1-ni/n)*nj*(1-nj/n))
	return rho

def pval(X, Na, Nb, Ntot):
    p=1
    if Na>1 and Nb>1:
        rv=hypergeom(M=Ntot, n=Na, N=Nb)
        p=1-rv.cdf(x=X-1)
    return p

def fdr(df_edges,threshold):
	sorted_pval=sorted(df_edges['pval'])
	u=threshold/df_edges.shape[0]
	check = []
	for i,pv in enumerate(sorted_pval):
	    i+=1
	    if pv>i*u:
	        FDR=i*u
	        #print(FDR5,i-2)
	        #check.append(i-2)
	        break
	return FDR

def save_pval_pairs(Dpval, name):
	f = open(name, 'w')
	f.write('source'+'\t'+'target'+'\t'+'pval'+'\t'+'weight'+'\n')
	for k,v in Dpval.items():
	    f.write(k[0]+'\t'+k[1]+'\t'+str(v[0])+'\t'+str(v[1])+'\n')
	f.close()
	return

def create_dtm(texts):
	vect = CountVectorizer(binary=True, min_df=2)#, max_df=0.7)
	X = vect.fit_transform(texts)
	dtm = pd.DataFrame(X.todense(), columns=vect.get_feature_names_out())
	#print(dtm.shape)
	return dtm

def collection_word_pairs(texts,dtm):
	#X = vect.fit_transform(texts)
	vocab = dtm.columns
	all_pair = set()
	for t in texts:
	    set_w = set(t.split()).intersection(vocab)
	    for w in combinations(set_w,2):
	        W=sorted(w)
	        all_pair.add((W[0],W[1]))
	#print(len(all_pair))

	return all_pair

def svn_fun(df_dtm, all_pair, name, method, threshold):

    Number_of_doc = df_dtm.shape[0]
    
    Dict_word = dict()
    for w in df_dtm.columns:
        w_count = sum(df_dtm.loc[:,w])
        Dict_word.update({w: w_count})
        
    Dict_pval=dict()
    for w1,w2 in tqdm(all_pair):
        X = (df_dtm.loc[:,[w1,w2]].sum(axis=1)==2).sum()
        if X>1:
            score = pval( X=X, Na =Dict_word[w1], Nb =Dict_word[w2], Ntot =Number_of_doc )
            weight = correlation(Dict_word[w1], Dict_word[w2], X, Number_of_doc)
            Dict_pval.update({(w1,w2): [score,weight]})

    save_pval_pairs(Dict_pval, name)
    df = pd.DataFrame.from_dict(Dict_pval,orient='index', columns=['pval','weight'])
    df.index = pd.MultiIndex.from_tuples(df.index, names=('source', 'target'))
    df = df.rename_axis(['source', 'target']).reset_index()
    # CHOOSE THRESHOLD
    if method=='bonf':
    	df = df[df['pval']<=threshold/df.shape[0]]
    if method=='fdr':
    	threshold_fdr = fdr(df,threshold)
    	df = df[df['pval']<=threshold_fdr]


    return df

def df_in_grahp(df, name):

	G = nx.from_pandas_edgelist(df,edge_attr='weight')
	mod_contr = []
	nodes = []
	n_topic = []
	#n_component = []
	#size_component = []

	topic_counter = 1
	for N, component in enumerate(sorted(nx.connected_components(G), key=len, reverse=True)):
		sub_g = G.subgraph(component)
		if sub_g.number_of_nodes()>4:
			partition_doc = nx.community.louvain_communities(sub_g, weight='weight')
			coms_doc = {}
			for n,set_w in enumerate(partition_doc):
				if len(set_w)>4:
					coms_doc.update({n:set_w})
					for w in set_w:
		
						sub_graph_comm = nx.subgraph(sub_g,set_w)
						ar = 1/(2*sub_g.number_of_edges())*sum([y for x,y in list(sub_graph_comm.degree())])
						q = 1/(2*sub_g.number_of_edges())*(sub_graph_comm.degree[w]-(sub_g.degree[w]*ar))
		
						mod_contr.append(q)
						nodes.append(w)
						n_topic.append('topic_'+str(topic_counter))
		                #n_component.append(N)
		                #size_component.append(len(component))
		            #print(topic_counter)
					topic_counter+=1

	df_community_partition = pd.DataFrame()
	#df_community_partition['graph_component'] = n_component
	#df_community_partition['size_component'] = size_component
	df_community_partition['topic'] = n_topic
	df_community_partition['word'] = nodes
	df_community_partition['modularity contribution'] = mod_contr
	# ADD STEMMING ROLE TO WORDS
	stemming_role = pd.read_csv('stemming_role.csv')
	final_community_partition = df_community_partition.merge(stemming_role,how='left',left_on='word',right_on='word')
	final_community_partition.sort_values(['topic','modularity contribution'],ascending=[True,False],inplace=True)
	final_community_partition.to_excel(name,index=False)

	return df_community_partition

def document_topic_overExpr(df_text, df_community_partition, threshold):
	topic_assigned = []
	#num_topic_assigned = []
	doc_id = []
	num_words_doc = []
	num_words_topic = []
	num_words_doc_topic = []
	all_words = []
	pvals = []
	correlations = []

	N_words = len(df_community_partition['word'])
	for j,row in df_text.iterrows():
	    
	    t = row['clean_text']
	    word_of_doc = set(t.replace(r'\.','').split()).intersection(df_community_partition['word'])
	    
	    for topic in set(df_community_partition['topic']):
	        words_of_topic = set(df_community_partition[df_community_partition['topic']==topic]['word'].tolist())

	        n_common_words = len(words_of_topic.intersection(word_of_doc))
	        if n_common_words>1:
	            pv = pval(X=n_common_words, Na=len(word_of_doc), Nb=len(words_of_topic), Ntot=N_words)
	            rho = correlation(ni=len(word_of_doc), nj=len(words_of_topic), nij=n_common_words, n=N_words)
	            pvals.append(pv)
	            correlations.append(rho)
	            topic_assigned.append(topic)
	            num_words_topic.append(len(words_of_topic))
	            num_words_doc_topic.append(n_common_words)
	            all_words.append(N_words)
	            num_words_doc.append(len(word_of_doc))
	            doc_id.append( row['text_id'] )
	df_doc_topic = pd.DataFrame()
	df_doc_topic['text_id'] = doc_id
	df_doc_topic['topic'] = topic_assigned
	#df_doc_topic['num_words_doc'] = num_words_doc
	#df_doc_topic['num_words_topic'] = num_words_topic
	#df_doc_topic['num_words_doc_and_topic'] = num_words_doc_topic
	#df_doc_topic['num_words_overall'] = all_words
	df_doc_topic['p-value'] = pvals
	df_doc_topic['correlation'] = correlations
	
	new_index = []
	for top in set(df_doc_topic['topic']):
		n_test = len(set(df_doc_topic[df_doc_topic['topic']==top]['text_id']))
		TRH = threshold/n_test
		ii = df_doc_topic[(df_doc_topic['topic']==top)&(df_doc_topic['p-value']<=TRH)].index
		new_index.extend(ii)
	df_doc_topic = df_doc_topic.loc[new_index,:]

	#num_topic_assigned = []
	#for d in df_doc_topic['text_id']:  
	#    num_topic_assigned.append( df_doc_topic[df_doc_topic['text_id']==d]['topic'].shape[0] )
	#df_doc_topic['number of topics'] = num_topic_assigned

	return df_doc_topic

def general_topic(dtm, df_text, df_doc_topic, df_community_partition, threshold):
	N_words = dtm.shape[1]
	svn_words = set(df_community_partition['word'])
	Pvals = []
	Correlations = []
	docs = []
	id_t = []
	n_t = []
	n_w_d = []
	n_w_t = []
	n_w_d_t = []
	n_w_tot = []
	for j,row in df_text.iterrows():
	    
	    t = row['clean_text']
	    word_of_doc = set(t.replace(r'\.','').split())
	    
	    n_common_words = len(svn_words.intersection(word_of_doc))
	    
	    if n_common_words>1:
	        
	        pv = pval(X=n_common_words, Na=len(word_of_doc), Nb=len(svn_words), Ntot=N_words)
	        rho = correlation(ni=len(word_of_doc), nj=len(svn_words), nij=n_common_words, n=N_words)
	        
	        Pvals.append(pv)
	        Correlations.append(rho)
	        docs.append(row['text_id'])
	        
	        id_t.append('topic_0')
	        
	        #if df_doc_topic[df_doc_topic['text_id']==row['text_id']].shape[0]>0:
	        #    n_t.append(df_doc_topic[df_doc_topic['text_id']==row['text_id']]['number of topics'].iloc[0])
	        #else:
	        #    n_t.append(0)
	        
	        n_w_d.append(len(word_of_doc))
	        n_w_t.append(len(svn_words))
	        n_w_d_t.append(n_common_words)
	        n_w_tot.append(N_words)

	df_topic_0 = pd.DataFrame()
	df_topic_0['text_id'] = docs
	df_topic_0['topic'] = id_t
	#df_topic_0['num_words_doc'] = n_w_d
	#df_topic_0['num_words_topic'] = n_w_t
	#df_topic_0['num_words_doc_and_topic'] = n_w_d_t
	#df_topic_0['num_words_overall'] = n_w_tot
	df_topic_0['p-value'] = Pvals
	df_topic_0['correlation'] = Correlations
	#df_topic_0['number of topics'] = n_t

	df_topic_0 = df_topic_0[df_topic_0['p-value']<=threshold/df_topic_0.shape[0]]

	return df_topic_0

def combine_df(df_doc_topic, df_topic_0, df_text, name):
	tot_df_doc_topic = pd.concat([df_doc_topic,df_topic_0])

	merge_df = tot_df_doc_topic.merge(df_text.loc[:,['text_id','Article title']],right_on='text_id',left_on='text_id')
	merge_df.reset_index(drop=True,inplace=True)
	merge_df.to_excel(name,index=False)

	return merge_df

def stats_topic(df_text, df_doc_topic, df_topic, label_topic):
	list_topics = []
	names_topics = []
	modularity = []
	n_words = []
	n_docs = []
	n_topj = []
	mean_cit = []
	mean_internal_cit = []
	mean_topic_cit = []

	for tp in set(label_topic['topic']):
	        
		list_topics.append(tp)
		docs = df_doc_topic[df_doc_topic['topic']==tp]['text_id'].tolist()
		n_docs.append(len(docs))
	    
		sub = df_text[df_text['text_id'].isin(docs)]
		n_topj.append(sub[sub['TOPJ']=='Y'].shape[0])
	    
		cit_list = sub['Total number of citations'].tolist()
		if len(cit_list)>0:
			mean_cit.append( np.mean(cit_list) )
		else:
			mean_cit.append(0)

		int_cit_list = sub['Number of internal citations'].tolist()
		if len(int_cit_list)>0:
			mean_internal_cit.append( np.mean(int_cit_list) )
		else:
			mean_internal_cit.append(0)

		doc_ref_topic = []
		for e in sub['References internal id']:
			if e!='':
				refs = set(docs).intersection(e.split())
				doc_ref_topic.append(len(refs))
		if len(doc_ref_topic)>0:
			mean_topic_cit.append(np.mean(doc_ref_topic))
		else:
			mean_topic_cit.append(0)

		names_topics.append(label_topic[label_topic['topic']==tp]['label'].iloc[0])
		mod = df_topic[df_topic['topic']==tp]['modularity contribution'].iloc[0]
		modularity.append(mod)
	    
		n_words.append(df_topic[df_topic['topic']==tp].shape[0])

	df_plotting = pd.DataFrame()
	df_plotting['Topic'] = list_topics
	df_plotting['Topic description'] = names_topics
	df_plotting['modularity contribution'] = modularity
	df_plotting['Number of words in topic'] = n_words
	df_plotting['Number of papers over-expressed'] = n_docs
	df_plotting['Number of papers from top journals over-expressed'] = n_topj
	df_plotting['Average number of citations'] = mean_cit
	df_plotting['Average number of citations within the dataset'] = mean_internal_cit
	df_plotting['Average number of citations within the topic'] = mean_topic_cit

	df_plotting.to_excel('stats_topic.xlsx',index=False)

	return df_plotting

def run_analysis(file_name, method='fdr', threshold=0.01):
	name1='SVN words.txt'
	name2='Topic definition.xlsx'
	name3='Topic Document association.xlsx'
	threshold_d=0.01
	threshold_0=0.01
# Import DataFrame (named 'df_text') with columns:
#						 'clean_text': preprocessed text as string
#						 'text_id': id associated to each document
#						 'tot_cit': total number of citations
#						 'internal_cit': number of citations within the dataset
#						 'TopJ': boolean if the pubblication belongs to Top Journal or not
#						 'references_internal_id': string of text_id separated by space (within the dataset) that cited the document
	df_text = pd.read_excel(file_name)

	txt = df_text['clean_text'].tolist()
	dtm = create_dtm(txt)
	all_pairs = collection_word_pairs(txt,dtm)
	print('Constructing the Statistically Validated Network..\n')
	df_edges = svn_fun(dtm, all_pairs, name1, method, threshold)#='svn_words.txt')
	print('Finding topics..\n')
	df_comm_part = df_in_grahp(df_edges, name2)#='topic_definition.xlsx')
	print('Calculating document-topic associations..\n')
	df_doc_topic = document_topic_overExpr(df_text, df_comm_part , threshold_d)#=0.01)
	df_topic_0 = general_topic(dtm, df_text, df_doc_topic, df_comm_part , threshold_0)#=0.01)
	merge_df = combine_df(df_doc_topic, df_topic_0, df_text, name3)#='Topic_Document_association.xlsx')
	#print('stats about words and topics..')
	#df_stats = stats_topic(merge_df, df_text, label_topic, name)

	return 

def run_stats(file_name1, file_name2, file_name3, file_name4):

	df_text = pd.read_excel(file_name1, na_filter=False)
	df_doc_topic = pd.read_excel(file_name2)
	df_topic = pd.read_excel(file_name3,)
	label_topic = pd.read_excel(file_name4)

	df_stats = stats_topic(df_text, df_doc_topic, df_topic, label_topic)

	return

def dataset_selection(file_name1, file_name2, file_name3, selection='broad'):
	df_text = pd.read_excel(file_name1, na_filter=False)
	df_doc_topic = pd.read_excel(file_name2, na_filter=False)
	label_topic = pd.read_excel(file_name3, na_filter=False)
	selected_topics = label_topic['topic'].to_list()
	selected_topics.append('topic_0')

	final_doc = df_doc_topic[df_doc_topic['topic'].isin(selected_topics)]['text_id'].to_list()
	df_text = df_text[df_text['text_id'].isin(final_doc)]

	if selection == 'broad':
		df_text = df_text[df_text['Total number of citations']>0]
	if selection == 'narrow':
		df_text = df_text[df_text['Number of internal citations']>0]

	#if topj:
	#	df_text = df_text[df_text['TOPJ']=='Y']

	df_text.to_excel('Final Dataset selection.xlsx',index=False)
	return 



def plot_stats_1(filename, name='topic_overview_1.pdf'):
	df = pd.read_excel(filename)
	df = df[df['Average number of citations'] != 0]
	df = df[df['Number of papers over-expressed'] != 0]

	Xax = df.loc[:,'Average number of citations within the dataset'].to_numpy()/df.loc[:, 'Average number of citations'].to_numpy()
	Yax = df.loc[:,'Number of papers from top journals over-expressed'].to_numpy()/df.loc[:,'Number of papers over-expressed'].to_numpy()
	mean_Y = np.mean(Yax)
	mean_X = np.mean(Xax)

	plt.figure(figsize=(8,8))
	plt.grid(True)#,alpha=0.5)

	sizes = df.loc[:,'Number of papers over-expressed']*10

	plt.scatter(Xax,Yax, s = sizes, alpha=0.4)#, c = colors, cmap = 'viridis')
	plt.axvline(x=mean_X)
	plt.axhline(y=mean_Y)


	#for n,i,j in zip(df['Topic'],Xax,Yax):
	#    if n in [9]:
	#        plt.text(x=i,y=j,s=str(n))
	#    else:
	#        plt.text(x=i+0.0005,y=j+0.001,s=str(n))#df_2.iloc[n,0]))

	plt.xlabel('Ratio citations')
	plt.ylabel('Ratio Top Journals')
	plt.title('Topic Overview')

	plt.savefig(name,dpi=300)#,transparent = True)
	#plt.show()

	return

def plot_stats_2(filename, name='topic_overview_2.pdf'):
	df = pd.read_excel(filename)
	Xax =df['Average number of citations within the dataset']
	Yax = df['Average number of citations']
	mean_Y = np.mean(Yax)
	mean_X = np.mean(Xax)

	plt.figure(figsize=(8,8))
	plt.grid(True)#,alpha=0.5)

	sizes = df.loc[:,'Number of papers over-expressed']*10

	plt.scatter(Xax, Yax, s = sizes, alpha=0.4)
	plt.axvline(x=mean_X)
	plt.axhline(y=mean_Y)


	for n,i,j in zip(df['Topic'],Xax,Yax):
	    if n in [9]:
	        plt.text(x=i,y=j,s=str(n))
	    else:
	        plt.text(x=i+0.0005,y=j+0.001,s=str(n))#df_2.iloc[n,0]))

	plt.xlabel('Average number of citations within the dataset')
	plt.ylabel('Average number of citations')
	plt.title('Topic Overview Citations')

	plt.savefig(name,dpi=300)#,transparent = True)
	#plt.show()

	return



def read_zootero_csv(filename):
	df = pd.read_csv(filename, na_filter=False)
	selected_columns = ['Author', 'Title', 'Abstract Note','Publication Title', 'Publication Year', 'Journal Abbreviation','References', 'DOI']
	df = df.loc[:,selected_columns]
	df.rename(columns={'Author':'First author','Title': 'Article title', 'Abstract Note': 'Abstracts','Publication Title': 'Source title', 'Publication Year':'Publication year',
		'Journal Abbreviation':'Journal abbreviation','DOI':'doi'}, inplace=True)

def read_scopus_csv(filename):
	df = pd.read_csv(filename, na_filter=False)
	selected_columns = ['Authors', 'Title', 'Abstract','Source title', 'Year', 'Abbreviated source title','References', 'Cited by','DOI']
	selected_columns = list(set(df.columns).intersection(selected_columns))
	df = df.loc[:,selected_columns]
	df.rename(columns={'Title': 'Article title', 'Abstract': 'Abstracts','Source Title': 'Source title', 'Year':'Publication year',
		'Cited by':'Total number of citations','Abbreviated source title':'Journal abbreviation','DOI':'doi'}, inplace=True)
	df['First author'] = [x.split(',')[0] for x in df['Authors']]

	print('Found '+str(df[df['Abstracts']==''].shape[0])+' empty abstracts')
	df = df[df['Abstracts']!='']
	shape1 = df.shape[0]
	df.drop_duplicates(['First author','Authors','Article title','Publication year'],inplace=True)
	df.reset_index(inplace=True, drop=True)
	df['text_id'] = ['d'+str(i) for i in range(df.shape[0])]

	print('The dataset has '+str(df.shape[0])+' documents')

	if shape1>df.shape[0]:
		print('removed '+str(shape1-df.shape[0])+' duplicated articles')

	print('Counting the citations internal the dataset for each document...\n')
	df = add_internal_citation_scopus(df)

	print('Saving the file Dataset_input.xlsx')
	writer = pd.ExcelWriter('Dataset_input.xlsx',
                        engine='xlsxwriter',
                        engine_kwargs={'options': {'strings_to_urls': False}})
	df.to_excel(writer,index=False)
	writer.close()
	#df.to_excel('Dataset_input.xlsx',index=False)
	return

def add_internal_citation_scopus(df):
	df['References internal id'] = ['']*df.shape[0]
	#df['References residual'] = ['']*df.shape[0]
	df['Article title'] = [unidecode(x) for x in df['Article title']]
	df['Authors'] = [unidecode(x) for x in df['Authors']]
	df['Source title'] = [x.replace('.','').replace(',','') for x in df['Authors']]
	cnt=0
	for j,row in tqdm(df.iterrows()):
	    cr = row['References']
	    all_temp = []
	    temp = []
	    temp_idx = []
	    for s in cr.split(';'):
	        try:
	            e = unidecode(s)
	            all_temp.append(e)
	            
	            au = []
	            tls = []
	            for path in e.split('.,'):
	            	if len(path)<20:
	            		au.append(path)
	            	else:
	            		tls.append(path)
	            #tl = re.search(r'\.,(.*)\(\d{4}\)',e).group(0)
	            #tl = tl.split('.,')[-1].split('(')[0]
	            #tl = re.sub(r'\s+\s+',' ',tl)
	            tl = tls[0].split(',')[0]
	            tl = re.sub(r'^ ','', tl)
	            tl = re.sub(r' $','', tl)
	            au = '.;'.join(au)+'.'
	            journal = tls[0].split(',')[1]
	            #escaped_string = re.escape(tl)
	            #pattern = re.compile('(.*)(?='+escaped_string+')')
	            #au = pattern.search(e).group(0)
	            #au = re.sub(r'\s+\s+',' ',au)
	            au = re.sub(r'^ ','', au)
	            au = re.sub(r' $','', au)

	            y = unidecode(re.search(r'\(\d{4}\)',e).group(0)).replace('(','').replace(')','')

	            #journal = e.split('('+y+')')[-1]
	            #journal = re.sub(r'^[^a-zA-Z]+','',journal)
	            if re.search('^pp',journal):
	                journal = journal.split(',')[1]
	            else:
	                journal = journal.split(',')[0]
	            journal = re.sub(r'\s+\s+',' ',journal)
	            journal = re.sub(r'^ ','', journal)
	            journal = re.sub(r' $','', journal)
	            journal = journal.replace('"','').replace("'",'')
	            journal = journal.replace('.','').replace(',','')
	            if df[df['Article title']==tl].shape[0]>0:
	                temp.append(e)
	                temp_idx.append(df[df['Article title']==tl]['text_id'].iloc[0])
	                cnt+=1
	                continue
	            if df[(df['Authors']==au)&(df['Publication year']==int(y))&(df['Source title']==journal)].shape[0]>0:
	                temp.append(e)
	                temp_idx.append(df[(df['Authors']==au)&(df['Publication year']==int(y))&(df['Source title']==journal)]['text_id'].iloc[0])
	                cnt+=1
	                continue
	        except:
	        	#cnt+=1
	        	continue

	    df.loc[j,'References internal id'] = ' '.join([str(x) for x in temp_idx])
	print(cnt)
	df['Number of internal citations']=[0]*df.shape[0]
	for j,row in df.iterrows():
	    id_cit = row['References internal id']
	    if id_cit!='':
	        for e in set(id_cit.split(' ')):
	            ii = df[df['text_id']==e].index[0]
	            df.loc[ii,'Number of internal citations']+=1
	return df


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
	            authors[j]= unidecode(c.replace('AF ','').lower())
	            for ii in range(n+1,n+4):
	                if cols[ii][:2]=='  ':
	                    authors[j]=authors[j]+';'+unidecode(cols[ii].replace('   ','').lower())
	                else:
	                    break
	            
	        if c[:2]=='AU':
	            first_author[j]= unidecode(c.replace('AU ','').lower())
	        if c[:2]=='TI':
	            article_title[j]= unidecode(c.replace('TI ','').lower())
	            
	            for ii in range(n+1,n+4):
	                if cols[ii][:2]=='  ':
	                    article_title[j]=article_title[j]+' '+cols[ii].replace('   ','').lower()
	                else:
	                    break
	                    
	        if c[:2]=='SO':
	            source_title[j]= unidecode(c.replace('SO ','').lower())
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
	print('Found '+str(len(ii))+' empty abstracts')
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
	shape1 = df_refs.shape[0]
	df_refs.drop_duplicates(['First author','Authors','Article title','Publication year'],inplace=True)
	df_refs.reset_index(inplace=True, drop=True)
	df_refs['text_id'] = ['d'+str(i) for i in range(df_refs.shape[0])]

	print('The dataset has '+str(df_refs.shape[0])+' documents')
	if shape1>df_refs.shape[0]:
		print('removed '+str(shape1-df_refs.shape[0])+' duplicated articles')

	print('Counting the citations internal the dataset for each document...\n')
	df_refs = add_internal_citation_wos(df_refs)
	print('Saving the file Dataset_input.xlsx')
	df_refs.to_excel('Dataset_input.xlsx',index=False)
	return


def add_top_journal(filename, df_file):
	topj_list = [x for x in open(filename).read().splitlines()]
	#j_list = [x.lower() for x in topj['JOURNAL TITLE']]
	df = pd.read_excel(df_file)
	df['TOPJ'] = ['N']*df.shape[0]
	df.loc[df[df['Source title'].isin(topj_list )].index,'TOPJ'] = 'Y'
	writer = pd.ExcelWriter('Dataset_input.xlsx',
                    engine='xlsxwriter',
                    engine_kwargs={'options': {'strings_to_urls': False}})
	df.to_excel(writer,index=False)
	writer.close()
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
						au = unidecode(e.split(',')[0].split()[0].replace(',',''))
						y = unidecode(e.split(',')[1].split()[0])
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

	other_stops.extend(['author', 'result', 'studi', 'research', 'effect', 'find', 'paper', 'provid', 'examin', 'develop', 'new', 'evalu',
		'implic', 'base', 'investig', 'categori', 'context', 'suggest', 'purpos', 'intent', 'previou', 'indic', 'contribut', 'publish',
		 'book', 'approach', 'method', 'analys', 'analysi', 'shed', 'light', 'abstract', 'science', 'summary', 'purpose', 'background',
		 'articl', 'amongst', 'conclusion', 'chapter', 'proceed'])

    
	text = re.sub(r"http\S+", " ", testo)
	text = re.sub(r'@\w+|＠\w+',' ',text)
	text = re.sub(r'©.*','',text)
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
	
	nltk.download('stopwords')
	print('Cleaning the abstracts..\n')
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
	print('Saving the file..')
	df.to_excel(filename,index=False)
	return
	    



