## Exploratory work with Papermill


This repository uses papermill to execute jupyter notebooks. 
To install this environment from scratch please use the following commands:
```bash
conda create --name myenv python=3.9
conda activate myenv
pip install -r requirements.txt
 ```
 To run the entire pipeline of notebooks please use the following commands
 ```bash
 python execute.py
 ```
 
 Papermill will automatically compile the notebooks, and the outputs can be directly observed in the /notebooks folder. 
I've written several notebooks to analyse the data, but the workflow goes as follows:
 
 -- notebooks --
 1. Preprocess the data - preprocess.ipynb
 2. Fit the data to a model - RandomForestCV.ipynb
 3. ""... MLP model" - mlp.ipynb
 4. Evaluate results - process_results.ipynb
 
 This workflow will run automatically via papermill. 
 
 -- results--
 This will contain the output benchmarking results from the notebooks.
 
 -- src --
 This contains the preprocessing function used for assembling the training and testing data. 

