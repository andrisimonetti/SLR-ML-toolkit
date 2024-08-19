

import numpy as np
import pandas as pd
import os

from collections import Counter, defaultdict
from itertools import combinations
from sklearn.feature_extraction.text import CountVectorizer
from scipy.stats import hypergeom
#import scipy


import networkx as nx
import community as commy

import time
from tqdm import tqdm


def correlation(ni,nj,nij,n):
    rho = (nij-(ni*nj)/n)/np.sqrt(ni*(1-ni/n)*nj*(1-nj/n))
    return rho

def pval(X, Na, Nb, Ntot):
    p=1
    if Na>1 and Nb>1:
        rv=hypergeom(M=Ntot, n=Na, N=Nb)
        p=1-rv.cdf(x=X-1)
    return p

def save_pval_pairs(Dpval, name='svn_words.txt'):

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

def svn_fun(df_dtm, all_pair,soglia=0.01, name):

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

  	save_pval_pairs(Dpval, name)
  	df = pd.DataFrame.from_dict(Dict_pval,orient='index', columns=['pval','weight'])
    df.index = pd.MultiIndex.from_tuples(df.index, names=('source', 'target'))
	df = df.rename_axis(['source', 'target']).reset_index()
    
	# CHOOSE THRESHOLD
	df = df[df['pval']<=soglia/df.shape[0]]
    return df

def df_in_grahp(df, name='topic_definition.xlsx'):

	G = nx.from_pandas_edgelist(df,edge_attr='weight')
	mod_contr = []
	nodes = []
	n_topic = []
	n_component = []
	size_component = []

	topic_counter = 1
	for N, component in enumerate(sorted(nx.connected_components(G), key=len, reverse=True)):
	    sub_g = G.subgraph(component)
	    if len(component)>=10:
	        partition_doc = commy.best_partition(sub_g, weight='weight')
	        coms_doc = defaultdict(list)
	        for k in partition_doc.keys():
	            coms_doc[partition_doc[k]].append(k)
	        if len(coms_doc)==1:
        		mod_contr.append(1)
                nodes.append(w)
                n_topic.append(topic_counter)
                n_component.append(N)
                size_component.append(len(component))
                topic_counter+=1
            else:
		        for c,set_w in coms_doc.items():
		            for w in set_w:

		                sub_graph_comm = nx.subgraph(sub_g,coms_doc[c])
		                ar = 1/(2*sub_g.number_of_edges())*sum([y for x,y in list(sub_graph_comm.degree())])
		                q = 1/(2*sub_g.number_of_edges())*(sub_graph_comm.degree[w]-(sub_g.degree[w]*ar))

		                mod_contr.append(q)
		                nodes.append(w)
		                n_topic.append(topic_counter)
		                n_component.append(N)
		                size_component.append(len(component))

	            topic_counter+=1

    df_community_partition = pd.DataFrame()
	df_community_partition['graph_component'] = n_component
	df_community_partition['size_component'] = size_component
	df_community_partition['topic'] = n_topic
	df_community_partition['word'] = nodes
	df_community_partition['modularity_contribution'] = mod_contr
	# ADD STEMMING ROLE TO WORDS
	df_community_partition.sort_values(['topic','modularity_contribution'],ascending=[True,False],inplace=True)
	df_community_partition.to_excel(name,index=False)

	return df_community_partition

def document_topic_overExpr(df_text, df_community_partition,soglia=0.01):
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
	df_doc_topic['num_words_doc'] = num_words_doc
	df_doc_topic['num_words_topic'] = num_words_topic
	df_doc_topic['num_words_doc_and_topic'] = num_words_doc_topic
	df_doc_topic['num_words_overall'] = all_words
	df_doc_topic['p-value'] = pvals
	df_doc_topic['correlation'] = correlations
	
	new_index = []
	for top in set(df_doc_topic['topic']):
	    if top in my_topic:
	        n_test = len(set(df_doc_topic[df_doc_topic['topic']==top]['text_id']))

	        SOGLIA = soglia/n_test #0.01/n_test # 0.05/n_test

	        ii = df_doc_topic[(df_doc_topic['topic']==top)&(df_doc_topic['p-value']<=soglia)].index
	        new_index.extend(ii)

	df_doc_topic = df_doc_topic.loc[new_index,:]
	return df_doc_topic

def general_topic(dtm, df_text, df_community_partition,soglia=0.01):
	N_words = dtm.shape[1]
	#svn_words = set(df_community_partition['word'])
	svn_words = set(df_community_partition[df_community_partition['topic']<27]['word'])#<27
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
	        
	        id_t.append(0)
	        
	        if final_doc_topic1[final_doc_topic1['text_id']==row['text_id']].shape[0]>0:
	            n_t.append(final_doc_topic1[final_doc_topic1['text_id']==row['text_id']]['number_of_topics'].iloc[0])
	        else:
	            n_t.append(0)
	        
	        n_w_d.append(len(word_of_doc))
	        n_w_t.append(len(svn_words))
	        n_w_d_t.append(n_common_words)
	        n_w_tot.append(N_words)

	df_topic_0 = pd.DataFrame()
	df_topic_0['text_id'] = docs
	df_topic_0['topic'] = id_t
	df_topic_0['num_words_doc'] = n_w_d
	df_topic_0['num_words_topic'] = n_w_t
	df_topic_0['num_words_doc_and_topic'] = n_w_d_t
	df_topic_0['num_words_overall'] = n_w_tot
	df_topic_0['p-value'] = Pvals
	df_topic_0['correlation'] = Correlations
	df_topic_0['number_of_topics'] = n_t

	SOGLIA = soglia
	df_topic_0 = df_topic_0[df_topic_0['p-value']<=SOGLIA/df_topic_0.shape[0]]

	return df_topic_0

def combine_df(df_doc_topic,df_topic_0,df_text,name='Topic_Document_association.xlsx'):
	tot_df_doc_topic = pd.concat([df_doc_topic,df_topic_0])
	#ADD INTERNAL/EXTERNAL CITATIONS ON "df_text"
	merge_df = tot_df_doc_topic.merge(df_text,right_on='text_id',left_on='text_id')
	merge_df.reset_index(drop=True,inplace=True)

	merge_df.to_excel(name,index=False)

	return merge_df


def run_analysis(file_name, name1, name2, name3):
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
	df_edges = svn_fun(dtm, all_pairs, soglia=0.01, name1)
	df_comm_part = df_in_grahp(df_edges, name1)#='topic_definition.xlsx')
	df_doc_topic = document_topic_overExpr(df_text, df_comm_part , soglia=0.01)
	df_topic_0 = general_topic(dtm, df_text, df_comm_part , soglia=0.01)
	merge_df = combine_df(df_doc_topic, df_topic_0, df_text, name2)#='Topic_Document_association.xlsx')

	print('numbers and stats about words and topics..')

	return 


###############################
#if __name__ == '__main__':
     # execute only if run as a script
#     main()









