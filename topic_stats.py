
import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt


def stats_topic(df_topic, df_text, label_topic, name):
	list_topics = []
	names_topics = []
	modularity = []
	n_words = []
	n_docs = []
	n_topj = []
	mean_cit = []
	mean_internal_cit = []
	mean_topic_cit = []

	for tp in set(df_topic['topic']):
	        
	    list_topics.append(tp)
	    docs = df_topic[df_topic['topic']==tp]['doc'].tolist()
	    n_docs.append(len(docs))
	    
	    sub = df_text[df_text['text_id'].isin(docs)]
	    n_topj.append(sub[sub['TOPJ']=='Y'].shape[0])
	    
	    mean_cit.append( np.mean(sub['tot_cit'].tolist()) )
	    mean_internal_cit.append( np.mean(sub['internal_cit'].tolist()) )

	    doc_ref_topic = []
	    for e in sub['references_internal_id']:
            refs = set(docs).intersection(e.split())
            doc_ref_topic.append(len(refs))
	    if len(doc_ref_topic)>0:
	        mean_topic_cit.append(np.mean(doc_ref_topic))
	    else:
	        mean_topic_cit.append(0)

        names_topics.append(label_topic[label_topic['topic']==tp]['label'].iloc[0])
        mod = label_topic[label_topic['topic']==tp]['modularity_contribution'].iloc[0]
        modularity.append(mod)
	    
	    n_words.append(df_topic[df_topic['topic']==tp]['num_words_topic'].iloc[0])

	df_plotting = pd.DataFrame()
	df_plotting['topic_id'] = list_topics
	df_plotting['label'] = names_topics
	df_plotting['modularity_contribution'] = modularity
	df_plotting['num_words_topic'] = n_words
	df_plotting['n_doc_over_expressed'] = n_docs
	df_plotting['n_doc_over_expressed_topj'] = n_topj
	df_plotting['average_n_cit'] = mean_cit
	df_plotting['average_n_cit_internal'] = mean_internal_cit
	df_plotting['average_n_cit_internal_topic'] = mean_topic_cit

	df_plotting.to_excel(name,index=False)

	return df_plotting


# STEP 2:
#
# Import DataFrame (named 'df_text') with columns:
#						 'clean_text': preprocessed text as string
#						 'text_id': id associated to each document
#						 'tot_cit': total number of citations
#						 'internal_cit': number of citations within the dataset
#						 'TopJ': boolean if the pubblication belongs to Top Journal or not
#						 'references_internal_id': string of text_id separated by space (within the dataset) that cited the document
#
# Import DataFrame (named 'df_label_topic') with columns:
#						 'topic_id': number indentifier for each topic - N.B. topic_id==0 must be topic 'General'
#						 'label': name of topic (excluded topic_id==0) given by researcher
#
# Import DataFrame (named 'df_topic') with columns:
#						 'text_id': id associated to each document
#						 'topic': id of topic
#						 'num_words_topic': number of words in each topic
#						 'modularity_contribution': modularity contribution of the topic
#						 'tot_cit': total number of citations
#						 'internal_cit': number of citations within the dataset
#						 'TopJ': boolean if the pubblication belongs to Top Journal or not
#						 'references_internal_id': string of text_id separated by space (within the dataset) that cited the document
def run_analysis(file_name1, file_name2, file_name3):

	df_text = pd.read_excel(file_name1)
	df_topic = pd.read_excel(file_name2)
	label_topic = pd.read_excel(file_name3)

	df_stats = stats_topic(df_topic, df_text, label_topic, name='stats_topic.xlsx')

	return 




