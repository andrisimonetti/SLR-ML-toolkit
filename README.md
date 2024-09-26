# SLR-toolkit
Machine Learning toolkit for selecting studies and topics in systematic literature reviews. The toolkit analyse the text of abstracts of a collection of papers through the network analysis, performing a topic analysis and providing the association between abstracts and topics.
Moreover, there are included optional procdures to manage files dowloaded from Scopus (in .csv format) or Web of Science (in .txt format) database, where the information about references must be included. To execute the toolkit a basic knowledge of python is required. 



## Setup
- Install Jupyter Notebook
- Install the following packages : `pandas`, `sklearn`, `scipy`, `networkx`, `community`, `matplotlib`, `itertools`, `collections`, `tqdm`
- Specfically, to install the package `community` (https://python-louvain.readthedocs.io/en/latest/api.html) run the following line from terminal:
    ```bash
    pip install python-louvain 
### Only for cleaning the texts:
 - Install the following packages : `spacy`, `nltk`, `unidecode`


## Download the files
- To download all files, run the following line from terminal:
   ```bash
   git clone https://github.com/andrisimonetti/SLR-toolkit_2.git


### STEP 0: Data creation (Optional)
0. In this Step you will use the following files `Dataset_creation.ipynb` and `wos_functions.py`, previously dowloaded.
1. Run the first step within the Notebook `Dataset_creation.ipynb`. By the functions within the module `wos_functions.py` you can organize the files in a data frame and count the references internal to the dataset for each document.
2. Run the second step within the Notebook `Dataset_creation.ipynb` to add a new column to the data frame to indicate which journal appears in the list of Top journals. If the file list is not provided by the user, the default list is provided by the file "TopJournal_list.txt"(https://journalranking.org).
3. In this Step you will use the following file `preprocessing.py`, previously dowloaded. Run the third step of the Notebook `Dataset_creation.ipynb` to process the text of abstacts. The preprocessing procedure to clean the texts consists of stemming the words and removing punctuations, stops-words and customized stop-words. To insert your list of stop-words..

### STEP 1: Analysis
0. In this Step you will use the following files `toolkit_functions.py`, `topic_stats.py` and `scopus_functions.py`, previously dowloaded.
1. Download the Notebook `Main Analysis.ipynb` and follow the routine described in. If you create the dataset by yourself, follow the instrunctions in the Notebook about the input files required.
2. By yourself select the topics from the file output of the previous step and assign them the labels. Then create a file excel to store the topic-label associations. The file  `topic_label_example.xlsx` is an example.
3. Run the second step within the Notebook `Main Analysis.ipynb` to create a file to store the statistics of topics. Follow the instrunctions in the Notebook about the input files required. 


   
### STEP2: Topic description 
1. Download the Notebook `Topic stats and plots.ipynb` and follow the routine described in. Follow the instrunctions about the input files required.


## The outputs:
After running the Notebook `Dataset_creation.ipynb` the outputs produced consist of 1 files: 
   - 'Dataset_input.xlsx': data frame collecting the informations about the documents


After running the Notebook `Main Analysis.ipynb` the outputs produced consist of 4 files: 
   - 'svn_words.txt': data frame describing the Statistically Validated Network; each row represents a link between two nodes ('source' and 'target') with its p-value and correlation coefficient.
   - 'topic_definition.xlsx': data frame describing the Topics found as community of words in the Statistically Validated Network;
   - 'Topic_Document_association.xlsx': data frame describing the associations between documents and topics; 'topic_0' represents the 'General'
 topic.
   - 'stats_topic.xlsx': data frame describing the topics and some related statistics. 


After running the Notebook `Topic stats and plots.ipynb` the outputs produced consist of 2 files:
   - 'topic_overview_1.pdf': scatter-plot of documents along two dimensions: 'ratio of citations' (x-axis) and 'ratio of top journals' (y-axis).
   - 'topic_overview_2.pdf': scatter-plot of documents along two dimensions: 'number of internal citations' (x-axis) and 'total number of citations' (y-axis).
