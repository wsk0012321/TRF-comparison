# Purpose
The repository contains codes and processed data for evaluating the influence of three parameters on the predictive power of a TRF model: scalar (linear vesus non-linear), word embedding methods (static versus dynamic) and similarity metrics (Pearson's correlation versus cosine similarity). Semantic dissimilarity values are used to predict EEG oscillations within the post-stimulus time window 300ms to 600ms. Our intention is to measure the methodoly introduced by Broderick et al. [1] and to postulate potential improvements. A non-linear TRF method was later introduced to solve both forward and backward models [2]. We will test whether TRF with non-linear scalars (log and poweer-law) outperfom the linear one.

## Data
If you want to replicate the whole procedure:
The original data of stimuli and EEG signals can be downloaded from CNSP workshops at https://datadryad.org/stash/dataset/doi:10.5061/dryad.070jc To train the word2vec model,  the BNC corpus is also required. You can have access to it at http://www.natcorp.ox.ac.uk/ Following the authors' original research, you will also need to download ukWaC and the English Wikipedia. (not recommended if you device has less than 192 GB of RAM)

It is highly recommended to import our preprocessed data and execute the in a refined workflow:
Semantic dissimilarity calculated and arranged can be found under the folder in formats .csv, .pkl and .mat. To perform TRF in Matlab, you can load the mat files directly. Notice that files of Run 9, 10, 16, 17 are removed because they contain words not captured in the word2vec model.

## Enviroment
for Python scripts: <br>
pip install -r requirements.txt

for R scripts: <br>
install.packages('dplyr') <br>
install.packages('ggplot2')

for Matlab scripts: <br>
Please download and install the mTRF Toolbox [3] and EEGLAB [4] <br>
Use addpath() function to register the scripts under the path evaluation/tools/

## Run (in refined way)
1. EEG data preparation: <br>
 (1). Download the raw EEG and stimuli data. <br>
 (2). Run the EEG_conversion.py in the folder eeg_preprocessing on the raw data. <br>
 (3). In Matlab, manually load every processed file of one run and remove files that contain excessive contamination with visual checking. <br>
 (4). Apply bandpass_filter.m to the data and interpolate contaminated channels manually with EEGLAB. Make sure that the value ids is equal to your number of loaded files. <br>
 (5). Use savefiles.m and specify the index of run.

2. Modelling and evaluation: <br>
 (1). In Matlab, set the variable eegPath to your EEG directory and set norma to 2 and nfold to 8. <br>
 (2). Run model_trf.m with four .mat files in the folder ./stimuli/ Please use the default parameters and modify the paths. <br>
 (3). Repeat the step 2 while changing the variable scale from 0 to 2. You will obtain 12 models. <br>

3. Statistic analysis: <br>
 (1). In R, run analysis.R on the outputs stored in the folder ./statistics/ <br>

[1] Broderick M P, Anderson A J, Di Liberto G M, et al. Electrophysiological correlates of semantic dissimilarity reflect the comprehension of natural, narrative speech[J]. Current Biology, 2018, 28(5): 803-809. e3. <br>
[2] Brodbeck C, Das P, Kulasingham J P, et al. Eelbrain: A Python toolkit for time-continuous analysis with temporal response functions. BioRxiv, 2021.08. 01.454687[EB/OL].(2021) <br>
[3] Crosse M J, Di Liberto G M, Bednar A, et al. The multivariate temporal response function (mTRF) toolbox: a MATLAB toolbox for relating neural signals to continuous stimuli[J]. Frontiers in human neuroscience, 2016, 10: 604. <br>
[4] Delorme A, Makeig S. EEGLAB: an open source toolbox for analysis of single-trial EEG dynamics including independent component analysis[J]. Journal of neuroscience methods, 2004, 134(1): 9-21.
