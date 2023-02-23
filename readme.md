# Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting

This folder concludes the code and data of our AGCRN model: [Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting](https://arxiv.org/pdf/2007.02842.pdf), which has been accepted to NeurIPS 2020. 

## Structure:

* data: including PEMSD4 and PEMSD8 dataset used in our experiments, which are released by and available at  [ASTGCN](https://github.com/Davidham3/ASTGCN/tree/master/data).

* lib: contains self-defined modules for our work, such as data loading, data pre-process, normalization, and evaluate metrics.

* model: implementation of our AGCRN model


## Requirements

Python 3.6.5, Pytorch 1.1.0, Numpy 1.16.3, argparse and configparser

## Create Virtual environment 

```
# creation environmen with name "my_venv"
virtualenv my_venv
or (if it does not work)
python3 -m venv my_venv

# activate
source my_venv/bin/activate

# install requirements
pip install -r requirements.txt

```

## Data preparation (using METR-LA and PEMS-BAY datasets)

### Download METR-LA and PEMS-BAY data from [Google Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) or [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g) links provided by [DCRNN](https://github.com/liyaguang/DCRNN).

### Preliminaries 

```
# Create data directories
mkdir -p data/METR-LA, data/PEMS-BAY

# inert npz files
Each npz file contains one key, named "data", the shape is (sequence_length, num_of_vertices, num_of_features).

# create the configuration file for each dataset (example of METR-LA)
[data]
    num_nodes = 207
    lag = 12
    horizon = 12
    val_ratio = 0.2
    test_ratio = 0.2
    tod = False
    normalizer = "std"
    column_wise = False
    default_graph = True

[model]
    input_dim = 1
    output_dim = 1
    embed_dim = 10
    rnn_units = 64
    num_layers = 2
    cheb_order = 2

[train]
    loss_func = "mask_mae"
    seed = 10
    batch_size = 64
    epochs = 100
    lr_init = 0.003
    lr_decay = False
    lr_decay_rate = 0.3
    lr_decay_step = 5, 20, 40, 70
    early_stop = True
    early_stop_patience = 15
    grad_norm = False
    max_grad_norm = 5
    real_value = False

[test]
    mae_thresh = None
    mape_thresh = 0.

[log]
    log_step = 40
    plot = False

```

### Train Commands

```
python3 model/Run.py
```

To replicate the results in PEMSD4 and PEMSD8 datasets, you can run the the codes in the "model" folder directly. If you want to use the model for your own datasets, please load your dataset by revising "load_dataset" in the "lib" folder and remember tuning the learning rate (gradient norm can be used to facilitate the training).

Please cite our work if you find useful.



