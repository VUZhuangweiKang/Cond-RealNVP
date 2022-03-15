# Traffic Anomaly Detection Via Conditional Normalizing Flow
This repo is created for reproducibility and sharing the codes used for the paper, Traffic Anomaly Detection Via Conditional
Normalizing Flow, submitted to The 25th IEEE International Conference on Intelligent Transportation Systems (IEEE ITSC 2022, under review).

## File Description
All code for the paper is in the form of python file and are in the root directory. File descriptions:

Configurations:
- model configuration file: [configs.json](configs.json)
- synthtetic test configuration file: [./config/syn_config.json](./config/syn_config.json)

Data:
- pre-process the INRIX dataset (private): [preprocess_inrix.py](./preprocess_inrix.py)
- generate synthetic testing set based on ground-truth data: [synthetic_gen.py](synthetic_gen.py)
- functions related to data operations (e.g, load and preprocess data): [load_data.py](./load_data.py)

Model:
- road segments clustering: [clustering.py](./clustering.py)
- LSTM Encoder Decoder model: [lstm_models.py](./lstm_models.py)
- conditional RealNVP: [CondRealNVP.py](./CondRealNVP.py)
- Multi-layer perceptron classification: [mlpclassifier.py](./mlpclassifier.py) 

## Run Experiments
1. Edit the configs.json and config/syn_config.json files to secify model and experiment configurations, then run
    ```shell
    python experiment.py
    ```
2. Trained model will be saved in the `experiments/Inrix/models/CondRealNVP/OPTICS/<cluster_id>` folder, evaluation metrics will be saved in `experiments/Inrix/logs/CondRealNVP/run1/OPTICS/<cluster_id>`.
3. Jupyter notebook for process and visualize experimental results is: [experiments/process.ipynb](./experiments/process.ipynb)

## Experimental results shown on submitted paper
Existing experimental results for individual clusters can be found [here](https://www.dropbox.com/sh/jf82je78ue0g98q/AADFa96whqcuPHS8l9X3JjwGa?dl=0). Please download, uncompress and the place the Inrix folder in `experiments`, then you can visualize the results using the same process.ipynb notebook.