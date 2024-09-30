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

- Step 1. Run the code in Phase 1 within the Notebook `Dataset creation and Analysis.ipynb`. In this Step you can organize the file dowloaded from Scopus or Web of Science in a data frame.
- Step 2. Run the code in Phase 2 within the Notebook `Dataset creation and Analysis.ipynb` to add a new column to the data frame to indicate which journal appears in the list of Top journals. If the file list is not provided by the user, the default list is provided by the file "TopJournal_list.txt"(https://journalranking.org).
- Step 3. Run the code in Phase 3 within the Notebook `Dataset creation and Analysis.ipynb` to process the text of abstacts. The preprocessing procedure to clean the texts consists of stemming the words and removing punctuations and stops-words.

#### Outputs:
- 'Dataset_input.xlsx': a data frame in which each row represents a paper and the columns are:
id of text; First author; Authors; Article title; Abstract; Source title; Pubblication year; Journal abbreviation; Journal iso abbreviation; References; Total number of citations; doi.


### STAGE 2 - Analysis
In this Step you will use the following files: `Dataset creation and Analysis.ipynb` and `toolkit_functions.py`.

- Step 1. Run the code in Phase 3 within the Notebook `Dataset creation and Analysis.ipynb`. In this Step you analyse the text of abstracts, defining topics and their association with the abstracts. At the end of this step you get 3 file outputs: `svn_words.txt`,`topic_definition.xlsx`, `Topic_Document_association.xlsx`. If you create the dataset by yourself (skipping Stage 1), follow the instrunctions in the Notebook about the input files required.
- Step 2. Select the most relevant topics from the file output `topic_definition.xlsx`, obtained in the previous step, and assign them the labels. Then, create a file excel with two columns that must be named 'topic' and 'label' to store the topic-label associations. The file  `topic_label_example.xlsx` is an example.
- Step 3. Run the code in Phase 4 within the Notebook `Dataset creation and Analysis.ipynb` to create a file to store the statistics of topics. Follow the instrunctions in the Notebook about the input files required.

#### Outputs:
- **svn_words.txt**: file with the edges list of the Statistically Validated Network of words; each row represents a link between two nodes ('source' and 'target') with its p-value and Pearson correlation coefficient. The file consists of four columns separated by '\t':
    - source, target, p-values, weight (Pearson correlation coefficient)
- **topic_definition.xlsx**: data frame describing the Topics found as community of words in the Statistically Validated Network. The data frame has the following columns:
    - 'topic': the identifier of the topic, e.g. 'topic_1'
    - 'word': the word in stemming form
    - 'modularity contribution': the importance of the node (word) within the community(topic) in terms of modularity contribution
    - 'original words': the list of words before the stemming procedure
- **Topic_Document_association.xlsx**: data frame describing the associations between documents and topics; 'topic_0' represents the 'General' topic.  The data frame has the following columns:
    - 'text_id': the identifier of the document, e.g. 'd1'
    - 'topic': the identifier of the topic, e.g. 'topic_1'
    - 'p-value': the p-value of the over-expression of the topic within the document
    - 'correlation': Person correlation coefficient of the over-expression of the topic within the document.

- 'svn_words.txt': data frame describing the Statistically Validated Network; each row represents a link between two nodes ('source' and 'target') with its p-value and correlation coefficient.
- 'topic_definition.xlsx': data frame describing the Topics found as community of words in the Statistically Validated Network;
- 'Topic_Document_association.xlsx': data frame describing the associations between documents and topics; 'topic_0' represents the 'General'  topic.
- 'stats_topic.xlsx': data frame describing the topics and some related statistics. 

   
### STAGE 3: Plots 
1. Download the Notebook `Plots.ipynb` and follow the routine described in. Follow the instrunctions about the input files required.

#### Outputs:
- 'topic_overview_1.pdf': scatter-plot of documents along two dimensions: 'ratio of citations' (x-axis) and 'ratio of top journals' (y-axis).
- 'topic_overview_2.pdf': scatter-plot of documents along two dimensions: 'number of internal citations' (x-axis) and 'total number of citations' (y-axis).
