# A Mention-Pair Coreference Resolution system
This is a mention-pair coreference resolution system implemented in Python for Python 3.7.2.
It takes as input files in CoNLL-2012/OntoNotes format and returns a file containing coreference annotations in OntoNotes format.
This system is statistical-based, i.e. it uses supervised learning algorithms to train a new model.
This system contains five different statistical models. 
These learning algorithms have been used for training the distinct models:
* Support Vector Machine
* Naive Bayes
* Perceptron
* Maximum Entropy
* Random Forest

11 features are used for training and prediction of coreference of mention-pairs.
To build coreference chains, closest-first clustering is used.


## Directory structure
This project contains the following directories with the following contents:
* root folder: 
  * other folders
  * Python scripts
  * Coreference Gold annotations
  * README.md (this file)
* Scores:
  * Output of Pradhan et al.'s (2014) scoring script of each model
* Predictions:
  * Coreference predictions of CoNLL-2012 of each model
* resources:
  * Bergsma & Lin's (2006) _Noun Gender and Number Data for Coreference Resolution_
* Trained_Models:
  * Binaries of each model


## Required libraries
To use the system, the following libraries have to be installed: *NLTK*, *WordNet*, *numpy*, *scikit-learn* and *imbalanced-learn*.


### Installing the libraries on a Unix environment (= Linux/MacOS)
To install *NLTK*, *numpy*, *scikit-learn* and *imbalanced-learn*, you have to run the following commands in the terminal:
```
sudo pip install -U nltk
sudo pip install -U numpy
sudo pip install -U scikit-learn
sudo pip install -U imbalanced-learn
```

To import *WordNet* to *NLTK*, you have to open Python in interactive mode (by typing `python` in the terminal) and execute the following comands:
```python
>>> import nltk
>>> nltk.download('wordnet')
```


### Installing the libraries on a Windows environment
For using *numpy* and *scikit-learn*, I would recommend installing *Anaconda* (https://www.anaconda.com/distribution/) as the libraries are included there.

Afterwards, you can install *imbalanced-learn* by running the following command in the terminal:
```
conda install -c conda-forge imbalanced-learn
```

To import *WordNet* to *NLTK*, you have to open Python in interactive mode (by typing `python` in the terminal) and execute the following comands:
```python
>>> import nltk
>>> nltk.download('wordnet')
```


## Train a model
To train a model, you have to change your working directory to the root folder of this project and execute the following command in the terminal:
```
python CoreferenceClassifier.py INPUTFILE/FOLDER OUTPUTFILE (CLASSIFIER)
```
INPUTFILE/FOLDER is the file or the folder containing the annotated documents in OntoNotes format.  
OUTPUTFILE is the location where the model in binary format should be saved.  
CLASSIFIER is optional. It is the algorithm that is used for learning (default=SVM).  
Possible values for Classifier:
  * NaiveBayes
  * Perceptron
  * SVM
  * MaxEnt
  * RandomForest


## Make Predictions 
To make coreference predictions for files in OntoNote-Format, you have to change your working directory to the root folder of this project and execute the following command in the terminal:
```
python CoreferenceResolution.py INPUTFILE OUTPUTFILE (CLASSIFIER_BINARY)
```
INPUTFILE/FOLDER is the file or the folder containing the annotated documents in OntoNotes format.  
OUTPUTFILE is the location where the predictions are saved in OntoNotes format.  
CLASSIFIER_BINARY is optional. It has to be the binary of an already trained model (default=SVM).  

This will also produce a file called *CONLL_GOLD_KEY* containing the gold annotations of the input files, which can be used for scoring the system.


## Evaluation Scores
For scoring the models, Pradhan et al.'s (2014) scoring script has been used.
The best model overall was a the Support Vector Machine with a recall of 48.99 %, a precision of 14.99 % and an F1-Score of 21.83 %, while the Naive Bayes model had a slightly better precision with 16.25 %.