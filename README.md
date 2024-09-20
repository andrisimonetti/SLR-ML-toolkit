# SLR-toolkit
Machine Learning toolkit for selecting studies and topics in systematic literature reviews.
The toolkit analyse a collection of abstracts..


## Setup
- Python >3
- Packages : `pandas`, `sklearn`, `scipy`, `networkx`, `community`, `matplotlib`, `itertools`, `collections`, `tqdm`
- To install the module `community` follow https://python-louvain.readthedocs.io/en/latest/api.html
### Optional for pre-processing text:
 - Packages : `spacy`, `nltk`, `unidecode`


## Usage
- Clone repository to download all files:
   ```bash
   git clone https://github.com/andrisimonetti/SLR-toolkit.git

### STEP 0: Data creation (Optional)
1. Dowload from Web of Science the dataset in .txt format.
2. Dowload `Import Dataset_creation.ipynb` and `wos_functions.py` files in the same folder. Run the Notebook `Dataset_creation.ipynb` with Jupyter. Through the functions within the module `wos_functions.py` you can organize the files in a pandas DataFrame and count the references internal to the dataset for each document.
3. Dowload the `preprocessing.py` file in the same folder of `Import Dataset_creation.ipynb`. Run the third step within the Notebook `Import Dataset_creation.ipynb`. The preprocessing consists of stemming the words and removing punctuations, stops-words and customized stop-words. To insert your list of stop-words..

### STEP 1: Analysis
1. Download `toolkit_functions.py`, `topic_stats.py` and `scopus_functions.py` files to import the functions needed.
2. Download and follow the routine described in the Notebook `Main Analysis.ipynb`. Follow the instrunctions about the input files required.

   
### STEP2: Topic description
1. Select the relevant topics and assign them the labels, and create a file excel. The file  `topic_label_example.xlsx` is an example.
2. Download and follow the routine described in the Notebook `Topic stats and plots.ipynb`. Follow the instrunctions about the input files required.


## Read the outputs:
After running the Notebook `Main Analysis.ipynb` the outputs produced consist of 3 files: 
   - 'svn_words.txt': data frame describing the Statistically Validated Network; each row represents a link between two nodes ('source' and 'target') with its p-value and correlation coefficient.
   - 'topic_definition.xlsx': data frame describing the Topics found as community of words in the Statistically Validated Network;
   - 'Topic_Document_association.xlsx': data frame describing the associations between documents and topics; 'topic_0' represents the 'General'
 topic.


After running the Notebook `Topic stats and plots.ipynb` the outputs produced consist of 3 files:
   - 'stats_topic.xlsx': data frame describing the topics and some related statistics.
   - 'topic_overview_1.pdf': scatter-plot of documents along two dimensions: 'ratio of citations' (x-axis) and 'ratio of top journals' (y-axis).
   - 'topic_overview_2.pdf': scatter-plot of documents along two dimensions: 'number of internal citations' (x-axis) and 'total number of citations' (y-axis).
