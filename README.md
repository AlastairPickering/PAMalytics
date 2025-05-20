# Pileated gibbon classifier
*Work in progress* model training and production implementation code for pileated gibbon classification from PAM recordings.

This repo contains the Jupyter notebook (**gibbon_classifier_train.ipynb**) used to train and evaluate a pileated gibbon classifier as well as separate folder with implementation scripts needed to take the trained classifier and embed it into a live workflow with results dashboard. <br> 

Note the underlying audio data is not provided in full due to size restrictions.<br>

The **gibbon_classifier_train.ipynb** training notebook incorporates reading in annotated json files, linking them to their audio files, applying data augmentation steps to increase positive class training sample size, audio preprocessing steps, transfer learning for feature extraction, and classification. Classification accuracy is assessed and the model exported for integration into the live workflow.

The **gibbon_classification_repo** folder contains the scripts and architecture needed to implement the trained classifier in a live environment. It has the following structure:<br>
- **audio:** the location for the audio files to be classified. Simply drop the unclassified files into this folder <br>
  - **processed:** folder for wav files that have been classifier. <br>
- **embeddings:** the individual features extracted from each audio file <br>
- **models:** the transformer used for feature extraction and the trained classifier exported from the training notebook (above) <br>
- **results:** a csv consolidating the classification results (prediction whether the file contains a gibbon call or not) (**classification_results.csv**) and the same results merged with more metadata for charting and analysis (**merged_classification_results.csv**) <br>
- **scripts:** the individual scripts needed to run the workflow. These include: <br>
  - **config.py** set up the environment and folder structure <br>
  - **preprocessing.py** which handles the audio preprocessing and feature extraction functions <br>
  - **pipeline.py** which manages the workflow steps and order <br>
  - **classify.py** which uses the saved classifier to predict whether the extracted features of each audio file contain a pileated gibbon or not <br>
  - **dashboard.py** which presents the results in an interactive streamlit dashboard app


  
