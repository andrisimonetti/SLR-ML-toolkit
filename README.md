# SLR-ML-toolkit
Machine Learning toolkit for selecting studies and topics in Systematic Literature Reviews. The toolkit analyses the text of abstracts of a collection of papers through network analysis and NLP tools, performing topic recognition and providing the association between abstracts and topics. Moreover, there are optional procedures to manage files downloaded from Scopus (in .csv format) or Web of Science (in .txt format) databases, provided with the information about references and number of citations. To execute the toolkit a basic knowledge of python is required.

The following procedure outlines four stages for setting up, creating, analyzing, and visualizing a dataset of articles extracting meaningful insights.
The process begins with setting up the necessary tools, either on a local machine (with Jupyter Notebook) or on Google Colab. Users can also experiment with provided data examples to familiarize themselves with the toolkit.

The first stage involves organizing and preparing the bibliographic data. This includes structuring data from sources like Scopus or Web of Science into a data frame, enhancing it with additional information about journal rankings, and pre-processing the abstracts.

Once the data is prepared, the analysis stage begins. The analysis focuses on identifying distinct topics within the abstracts through the application of the Statistically Validated Networks method to build a network of words. Then, the toolkit identifies topics and their associations with the documents. Users can select and assign labels to the most relevant topics. Additionally, topic-related statistics are computed and saved for further examination, providing deeper insights into the distribution and significance of various themes in the literature.

Once the selection of topics is provided, users can apply filters based on citation counts and top journal appearances to create a final selection of articles.

Finally, visualizations are created to present the findings. Scatterplots illustrate key metrics such as citation ratios and the representation of top journals within the dataset. These visual outputs offer a clear overview of the selected topics and their relevance, enabling more informed exploration of the dataset.

Overall, the entire process outputs structured files that include the cleaned dataset, topic definitions, document-topic associations, visual representations of the results, and the selection of articles. 


### STAGE 0 - Set up
1. If you use **Jupyter Notebook** on your computer: Download the files `Dataset creation and Analysis.ipynb` and `toolkit_functions.py` in the same folder.

	Packages for the analysis: `numpy`,`pandas`, `sklearn`, `scipy`, `networkx`, `matplotlib`, `tqdm`

	Only for cleaning the texts: `spacy`, `nltk`, `unidecode`


2. If you use **Google Colab**: At this link https://colab.google click on the button 'Open Colab'. Then, click on 'GitHub' and paste the link https://github.com/andrisimonetti/SLR-ML-toolkit to load the Notebook `Dataset creation and Analysis.ipynb`. Download the file `toolkit_functions.py` from this repository. Then, click on the folder shape on the left panel and load the file `toolkit_functions.py` and your downloaded file from Scopus or Web of Science.


3. You can try the toolkit with the data examples provided in the folder 'Data'.



### STAGE 1 - Data creation
- Step 0. Open the Notebook `Dataset creation and Analysis.ipynb` and run the codes in STAGE 0 to install the packages not already installed and to import the functions.
- Step 1. Run the code in Phase 1 within the Notebook `Dataset creation and Analysis.ipynb`. In this Step, you can organize the downloaded file from Scopus or Web of Science in a data frame.
- Step 2. Run the code in Phase 2 within the Notebook `Dataset creation and Analysis.ipynb` to add a new column to the data frame to indicate which journal appears in a list of Top Journals. If the file list is not provided by the user, the default list is provided by the file "TopJournal_list.txt" (reference:https://journalranking.org). To use the file, it must be in the same folder of the Notebook `Dataset creation and Analysis.ipynb`. You can find the file in the folder 'Data'.
- Step 3. Run the code in Phase 3 within the Notebook `Dataset creation and Analysis.ipynb` to process the text of the abstracts. The pre—processing procedure to clean the texts consists of stemming the words and removing punctuations and stops-words.

#### Outputs:
- **Dataset_input.xlsx**: a data frame in which each row represents a paper, and the columns are:
text id; First author; Authors; Article title; Abstract; Source title; Publication year; Journal abbreviation; Journal ISO abbreviation; References; Total number of citations; DOI.


### STAGE 2 - Analysis
- Step 0. Open the Notebook `Dataset creation and Analysis.ipynb`. If it is not already open, import the module `toolkit_functions`.
- Step 1. Run the code in Phase 3 within the Notebook `Dataset creation and Analysis.ipynb`. In this Step, you can analyse the text of the abstracts, defining topics and their association with the abstracts. At the end of this Step, you will get 3 file outputs: `SVN words.txt`,`Topic Definition.xlsx`, `Topic Document association.xlsx`. If you create the dataset by yourself (skipping Stage 1), follow the instructions in the Notebook about the input files required in Phase 3.
- Step 2. Select the most relevant topics from the file output `topic_definition.xlsx`, obtained in the previous step, and assign them the labels. Then, create an excel file with two columns that must be named 'topic' and 'label' to store the topic-label associations. Then, to use the file, it must be in the same folder of the Notebook `Dataset creation and Analysis.ipynb`. The file `topic_label_example.xlsx` is an example, and you can find it in the folder 'Data'. 
- Step 3. Run the code in Phase 4 within the Notebook `Dataset creation and Analysis.ipynb` to create a file to store the statistics of the selected topics. Follow the instructions in the Notebook about the input files required.
- Step 4. Run the code in Phase 5 within the Notebook `Dataset creation and Analysis.ipynb` to create a file with the final selection of articles. For the final selection, you can apply two filters: one related to the number of citations, choosing either 'broad' (at least one citation in Scopus or Web of Science database) or 'narrow' (at least one citation within the downloaded dataset). The second filter allows you to select whether the articles in the final selection must be published in journals that appear in the "TopJournal_list.txt" file. At the end of this Step, you will get a file named `Final dataset selection.xlsx` containing only the articles that meet your filers.

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
    - 'Article title': the title of the paper.
    - 'topic': the identifier of the topic, e.g. 'topic_1'.
    - 'p-value': the p-value of the over—expression of the topic within the document.
    - 'correlation': Pearson correlation coefficient of the over—expression of the topic within the document.
- **Final Dataset selection.xlsx**: data frame containing only the articles in the final selection that meet your filers.

   
### STAGE 3: Plots
- Step 1. Run the code in Phase 6 within the Notebook `Dataset creation and Analysis.ipynb` to save the scatterplots of topics along different dimensions. Follow the instructions in the Notebook about the input files required.

#### Outputs:
- **topic_overview_1.pdf**: scatterplot of topics along two dimensions: 'ratio of citations' (x-axis) and 'ratio of top journals' (y-axis). 'ratio of citations' is obtained as the 'Average number of citations within the dataset' divided by 'Average number of citations'. 'ratio of top journals' is obtained as the 'Number of papers from top journals over—expressed' divided by 'Number of papers over—expressed'.
- **topic_overview_2.pdf**: scatterplot of topics along two dimensions: 'Average number of citations within the dataset' (x-axis) and 'Average number of citations' (y-axis).
