# SLR-ML-toolkit
Machine Learning toolkit for selecting studies and topics in Systematic Literature Reviews. The toolkit analyses the text of abstracts of a collection of papers through network analysis and NLP tools, performing topic recognition and providing the association between abstracts and topics.
Moreover, there are optional procedures to manage files downloaded from Scopus (in .csv format) or Web of Science (in .txt format) databases with the information about references and number of citations. To execute the toolkit a basic knowledge of python is required.

The following procedure outlines the steps for setting up, creating, analyzing, and visualizing a dataset of articles to extract meaningful insights.
The process begins with setting up the necessary tools, either on a local machine or on Google Colab, where required files and packages are installed. Users can also experiment with provided sample data to familiarize themselves with the toolkit.

The first key step involves organizing and preparing the bibliographic data. This includes structuring data from sources like Scopus or Web of Science into a dataframe, enhancing it with additional information about journal rankings, and pre-processing the abstracts. Text cleaning is performed through stemming and the removal of stopwords, punctuation, and other irrelevant elements, ensuring the dataset is ready for analysis.

The core of the analysis focuses on identifying distinct topics within the abstracts through a statistical examination of word networks. Topics are uncovered, and users can assign labels to the most relevant ones, generating associations between the topics and specific documents. This analysis also includes the computation of topic-related statistics, providing deeper insights into the distribution and significance of various themes in the literature.

Finally, visualizations are created to present the findings. Scatterplots illustrate key metrics such as citation ratios and the representation of top journals within the dataset. These visual outputs offer a clear overview of the identified topics and their relevance, enabling more informed exploration of the dataset.

Overall, the procedure produces a range of outputs, including the cleaned dataset, detailed topic definitions, document-topic associations, and visual representations of the results.


### STAGE 0 - Set up
1. If you use **Jupyter Notebook** on your computer: Download the files `Dataset creation and Analysis.ipynb` and `toolkit_functions.py` in the same folder.

	Packages for the analysis: `numpy`,`pandas`, `sklearn`, `scipy`, `networkx`, `matplotlib`, `tqdm`

	Only for cleaning the texts: `spacy`, `nltk`, `unidecode`


2. If you use **Google Colab**: At this link https://colab.google click on the button 'Open Colab'. Then, click on 'GitHub' and paste the link https://github.com/andrisimonetti/SLR-ML-toolkit to load the Notebook `Dataset creation and Analysis.ipynb`. Download the file `toolkit_functions.py` from this repository. Then, click on the folder shape on the left panel and load the file `toolkit_functions.py` and your downloaded file from Scopus or Web of Science.


3. You can try the toolkit with the data examples provided in the folder 'data'.



### STAGE 1 - Data creation
- Step 0. Open the Notebook `Dataset creation and Analysis.ipynb`.
- Step 1. Run the code in Phase 1 within the Notebook `Dataset creation and Analysis.ipynb`. In this Step you can organize the downloaded file from Scopus or Web of Science in a data frame.
- Step 2. Run the code in Phase 2 within the Notebook `Dataset creation and Analysis.ipynb` to add a new column to the data frame to indicate which journal appears in a list of Top Journals. If the file list is not provided by the user, the default list is provided by the file "TopJournal_list.txt" (reference:https://journalranking.org).
- Step 3. Run the code in Phase 3 within the Notebook `Dataset creation and Analysis.ipynb` to process the text of the abstracts. The pre—processing procedure to clean the texts consists of stemming the words and removing punctuations and stops-words.

#### Outputs:
- **Dataset_input.xlsx**: a data frame in which each row represents a paper, and the columns are:
text id; First author; Authors; Article title; Abstract; Source title; Publication year; Journal abbreviation; Journal ISO abbreviation; References; Total number of citations; DOI.


### STAGE 2 - Analysis
- Step 0. Open the Notebook `Dataset creation and Analysis.ipynb`, if not yet.
- Step 1. Run the code in Phase 3 within the Notebook `Dataset creation and Analysis.ipynb`. In this Step you can analyse the text of the abstracts, defining topics and their association with the abstracts. At the end of this step you get 3 file outputs: `SVN words.txt`,`Topic Definition.xlsx`, `Topic Document association.xlsx`. If you create the dataset by yourself (skipping Stage 1), follow the instructions in the Notebook about the input files required in Phase 3.
- Step 2. Select the most relevant topics from the file output `topic_definition.xlsx`, obtained in the previous step, and assign them the labels. Then, create an excel file with two columns that must be named 'topic' and 'label' to store the topic-label associations. The file  `topic_label_example.xlsx` is an example.
- Step 3. Run the code in Phase 4 within the Notebook `Dataset creation and Analysis.ipynb` to create a file to store the statistics of the selected topics. Follow the instructions in the Notebook about the input files required.

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
- Step 1. Run the code in Phase 5 within the Notebook `Dataset creation and Analysis.ipynb` to save the scatterplots of topics along different dimensions. Follow the instructions in the Notebook about the input files required.

#### Outputs:
- **topic_overview_1.pdf**: scatterplot of topics along two dimensions: 'ratio of citations' (x-axis) and 'ratio of top journals' (y-axis). 'ratio of citations' is obtained as the 'Average number of citations within the dataset' divided by 'Average number of citations'. 'ratio of top journals' is obtained as the 'Number of papers from top journals over—expressed' divided by 'Number of papers over—expressed'.
- **topic_overview_2.pdf**: scatterplot of topics along two dimensions: 'Average number of citations within the dataset' (x-axis) and 'Average number of citations' (y-axis).
