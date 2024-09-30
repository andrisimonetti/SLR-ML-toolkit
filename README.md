# SLR-toolkit
Machine Learning toolkit for selecting studies and topics in systematic literature reviews. The toolkit analyse the text of abstracts of a collection of papers through the network analysis, performing a topic analysis and providing the association between abstracts and topics.
Moreover, there are included optional procdures to manage files dowloaded from Scopus (in .csv format) or Web of Science (in .txt format) database, where the information about references must be included. To execute the toolkit a basic knowledge of python is required. 



## Python setup: 
- Install the following packages : `pandas`, `sklearn`, `scipy`, `networkx`, `community`, `matplotlib`, `itertools`, `collections`, `tqdm`
- Specifically, to install the package `community` (https://python-louvain.readthedocs.io/en/latest/api.html) run the following line from terminal:
    ```bash
    pip install python-louvain 
#### Only for cleaning the texts:
 - Install the following packages : `spacy`, `nltk`, `unidecode`


## Download the files
- Dowload all the files in the same folder
  
or
  
- Run the following line from terminal to create a folder with all files in:
   ```bash
   git clone https://github.com/andrisimonetti/SLR-toolkit_2.git


### STAGE 1 - Data creation
In this Stage you will use the following files: `Dataset creation and Analysis.ipynb` and `preprocessing_functions.py`.

- Step 1. Run the code in Phase 1 within the Notebook `Dataset_creation.ipynb`. In this Step you can organize a file, dowload in a data frame and count the references internal to the dataset (downloaded) for each document.
- Step 2. Run the code in Phase 2 within the Notebook `Dataset_creation.ipynb` to add a new column to the data frame to indicate which journal appears in the list of Top journals. If the file list is not provided by the user, the default list is provided by the file "TopJournal_list.txt"(https://journalranking.org).
- Step 3. In this Step you will use the following file `preprocessing.py`, previously dowloaded. Run the code in Phase 3 within the Notebook `Dataset_creation.ipynb` to process the text of abstacts. The preprocessing procedure to clean the texts consists of stemming the words and removing punctuations and stops-words and customized stop-words. To insert your list of stop-words..
- (Optional Step) If you create the dataset by yourself, follow the instrunctions in the Notebook about the input files required.

#### Outputs:
- 1 files named 'Dataset_input.xlsx'; a DataFrame in which each row represents a paper and the columns are:
id of text; First author; Authors; Article title; Abstract; Source title; Pubblication year; Journal abbreviation; Journal iso abbreviation; Feferences; Number of citations; doi.


### STAGE 2 - Analysis
- Step 0. In this Step you will use the following files `Main Analysis.ipynb`, `toolkit_functions.py`, `topic_stats.py` and `scopus_functions.py`, previously downloaded.
- Step 1. Run the code in Phase 1 within the Notebook `Main Analysis.ipynb`. At the end of this step you will get 3 file outputs: `svn_words.txt`,`topic_definition.xlsx`, `Topic_Document_association.xlsx`.
- Step 2. By yourself select the topics from the file output `topic_definition.xlsx` obtained in the previous step and assign them the labels. Then, create a file excel with two columns that must be named 'topic' and 'label' to store the topic-label associations. The file  `topic_label_example.xlsx` is an example.
- Step 3. Run the code in Phase 2 within the Notebook `Main Analysis.ipynb` to create a file to store the statistics of topics. Follow the instrunctions in the Notebook about the input files required. 

#### Outputs:
- 'svn_words.txt': data frame describing the Statistically Validated Network; each row represents a link between two nodes ('source' and 'target') with its p-value and correlation coefficient.
- 'topic_definition.xlsx': data frame describing the Topics found as community of words in the Statistically Validated Network;
- 'Topic_Document_association.xlsx': data frame describing the associations between documents and topics; 'topic_0' represents the 'General'  topic.
- 'stats_topic.xlsx': data frame describing the topics and some related statistics. 

   
### STAGE 3: Plots 
1. Download the Notebook `Plots.ipynb` and follow the routine described in. Follow the instrunctions about the input files required.

#### Outputs:
- 'topic_overview_1.pdf': scatter-plot of documents along two dimensions: 'ratio of citations' (x-axis) and 'ratio of top journals' (y-axis).
- 'topic_overview_2.pdf': scatter-plot of documents along two dimensions: 'number of internal citations' (x-axis) and 'total number of citations' (y-axis).
