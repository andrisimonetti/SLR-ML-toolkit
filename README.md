# SLR-toolkit
Machine Learning toolkit for selecting studies and topics in systematic literature reviews


## Setup
- Python >3
- Packages : `pandas`, `sklearn`, `scipy`, `networkx`, `community`, `matplotlib`, `itertools`, `collections`
- To install the module `community` follow https://python-louvain.readthedocs.io/en/latest/api.html


## Usage

### STEP1
- Clone repository:
   ```bash
   git clone https://github.com/andrisimonetti/SLR-toolkit.git

#### or
1. Download `toolkit_functions.py` to import the functions needed.
2. Download and run the Notebook `Main Analysis.ipynb` following the instrunctions about the input file required.
   
### STEP2
1. Select the relevant topics and assign them the labels: create file excel.
2. Download and run the Notebook `Topic stats and plots.ipynb` following the instrunctions about the input file required.


## Read the outputs:
After running the Notebook `Main Analysis.ipynb` the outputs produced consist of 3 files: 
   - 'svn_words.txt': data frame describing the Statistically Validated Network; each row represent a link with its p-value and correlation coefficient.
   - 'topic_definition.xlsx': data frame describing the Topics found as community of words in the Statistically Validated Network;
   - 'Topic_Document_association.xlsx': data frame describing the associations between documents and topics; Topic id 0 represents the 'General'
 topic.
