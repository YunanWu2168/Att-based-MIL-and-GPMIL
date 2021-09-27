# Phase 2: VGPMIL model classification 
This repository holds the code to train VGPMIL on CNN features
We use CNN features as the input for VGPMIL.

# How to use
* Navigate to the folder 'cd Att-based-MIL-and-GPMIL/phase_2_vgpmil/'.
* To install all requirements please use anaconda/miniconda and type the following commands:
    * conda env create -f environment.yaml
    * conda activate cnn_plus_vgpmil
* Run the program :
    * python src/main.py
* To change parameters, please use the 'config.yaml'
* The metrics will be stored in the output folder (defined in the config.yaml)

# Pre-extracted Features
* In the folder 'Att-based-MIL-and-GPMIL/phase_2_vgpmil/data/' the extracted features are provided to run only the 
second phase for final results
* All provided features have 8 dimensions as this is the best performing version (see the paper)