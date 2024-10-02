# SLR-ML-toolkit
Machine Learning toolkit for selecting studies and topics in Systematic Literature Reviews. The toolkit analyses the text of abstracts of a collection of papers through network analysis, performing a topic analysis and providing the association between abstracts and topics.
Moreover, there are optional procedures to manage files dowloaded from Scopus (in .csv format) or Web of Science (in .txt format) databases with the information about references and number of citations. To execute the toolkit a basic knowledge of python is required.


### STAGE 0 - Set up
1. If you use **Jupyter Notebook** on your computer: Download the files `Dataset creation and Analysis.ipynb` and `toolkit_functions.py` in the same folder.

	Packages for the analysis: `numpy`,`pandas`, `sklearn`, `scipy`, `networkx`, `matplotlib`, `tqdm`

	Only for cleaning the texts: `spacy`, `nltk`, `unidecode`


2. If you use **Google Colab**: At this link https://colab.google click on the button 'Open Colab'. Then, click on 'GitHub' and paste the link https://github.com/andrisimonetti/SLR-ML-toolkit to load the Notebook `Dataset creation and Analysis.ipynb`. Download the file `toolkit_functions.py` from this repository. Then, click on the folder shape on the left panel and load the file `toolkit_functions.py` and your downloaded file from Scopus or Web of Science.


3. You can try the toolkit with the data examples provided in the folder 'data'.



### STAGE 1 - Data creation
- Step 0. Open the Notebook `Dataset creation and Analysis.ipynb`.
- Step 1. Run the code in Phase 1 within the Notebook `Dataset creation and Analysis.ipynb`. In this Step you can organize the dowloaded file from Scopus or Web of Science in a data frame.
- Step 2. Run the code in Phase 2 within the Notebook `Dataset creation and Analysis.ipynb` to add a new column to the data frame to indicate which journal appears in a list of Top Journals. If the file list is not provided by the user, the default list is provided by the file "TopJournal_list.txt" (reference:https://journalranking.org).
- Step 3. Run the code in Phase 3 within the Notebook `Dataset creation and Analysis.ipynb` to process the text of the abstacts. The pre—processing procedure to clean the texts consists of stemming the words and removing punctuations and stops-words.

#### Outputs:
- **Dataset_input.xlsx**: a data frame in which each row represents a paper and the columns are:
text id; First author; Authors; Article title; Abstract; Source title; Pubblication year; Journal abbreviation; Journal ISO abbreviation; References; Total number of citations; DOI.


### STAGE 2 - Analysis
- Step 0. Open the Notebook `Dataset creation and Analysis.ipynb`, if not yet.
- Step 1. Run the code in Phase 3 within the Notebook `Dataset creation and Analysis.ipynb`. In this Step you analyse the text of the abstracts, defining topics and their association with the abstracts. At the end of this step you get 3 file outputs: `SVN words.txt`,`Topic Definition.xlsx`, `Topic Document association.xlsx`. If you create the dataset by yourself (skipping Stage 1), follow the instrunctions in the Notebook about the input files required in Phase 3.
- Step 2. Select the most relevant topics from the file output `topic_definition.xlsx`, obtained in the previous step, and assign them the labels. Then, create an excel file with two columns that must be named 'topic' and 'label' to store the topic-label associations. The file  `topic_label_example.xlsx` is an example.
- Step 3. Run the code in Phase 4 within the Notebook `Dataset creation and Analysis.ipynb` to create a file to store the statistics of the selected topics. Follow the instrunctions in the Notebook about the input files required.

#### Outputs:
- **SVN words.txt**: file with the edges list of the Statistically Validated Network of words; each row represents a link between two nodes ('source' and 'target') with its p-value and Pearson correlation coefficient. The file consists of four columns separated by '\t':
    - source, target, p-values, weight (Pearson correlation coefficient).
- **Topic definition.xlsx**: data frame describing the Topics found as community of words in the Statistically Validated Network. The data frame has the following columns:
    - 'topic': the identifier of the topic, e.g. 'topic_1'.
    - 'word': the word in stemming form.
    - 'modularity contribution': the importance of the node (word) within the community (topic) in terms of modularity contribution.
    - 'original words': the list of words before the stemming procedure.
- **Topic Document association.xlsx**: data frame describing the associations between documents and topics; 'topic_0' represents the topic named 'General'.  The data frame has the following columns:
    - 'text_id': the identifier of the document, e.g. 'd1'.
    - 'topic': the identifier of the topic, e.g. 'topic_1'.
    - 'p-value': the p-value of the over—expression of the topic within the document.
    - 'correlation': Pearson correlation coefficient of the over—expression of the topic within the document.

   
### STAGE 3: Plots
In this Step you will use the following files: `Dataset creation and Analysis.ipynb` and `toolkit_functions.py`.
- Step 1. Run the code in Phase 5 within the Notebook `Dataset creation and Analysis.ipynb` to save the scatter-plots of topics along different dimensions. Follow the instrunctions in the Notebook about the input files required.

#### Outputs:
- **topic_overview_1.pdf**: scatter-plot of topics along two dimensions: 'ratio of citations' (x-axis) and 'ratio of top journals' (y-axis). 'ratio of citations' is obtained as the 'Average number of citations within the dataset' divided by 'Average number of citations'. 'ratio of top journals' is obtained as the 'Number of papers from top journals over—expressed' divided by 'Number of papers over—expressed'.
- **topic_overview_2.pdf**: scatter-plot of topics along two dimensions: 'Average number of citations within the dataset' (x-axis) and 'Average number of citations' (y-axis).
