# Stage-II: SDCS Training Script

This repository contains a training script for the SDCS Stage-II model using PyTorch.

## Table of Contents

- [Stage-II: SDCS Training Script](#stage-ii-sdcs-training-script)
  - [Table of Contents](#table-of-contents)
  - [Dataset Organization](#dataset-organization)
  - [Setting Up the Training Environment](#setting-up-the-training-environment)
  - [Running the Training and Testing](#running-the-training-and-testing)
    - [Command Line Arguments](#command-line-arguments)
    - [Testing the Model](#testing-the-model)
  - [License](#license)
  - [Contact](#contact)

## Dataset Organization

The dataset should be organized in the following structure:

```
data
|-- CIA
|   |-- CSBS
|   |   |-- MOD09GA_A0000000.sur_reflCSBS.tif
|   |   |-- MOD09GA_A2001290.sur_reflCSBS.tif
|   |   `-- ...
|   |-- Landsat
|   |   |-- L71093084_08420011007_HRF_modtran_surf_ref_agd66.tif
|   |   |-- L71093084_08420011016_HRF_modtran_surf_ref_agd66.tif
|   |   `-- ...
|   |-- MODIS
|       |-- MOD09GA_A2001281.sur_refl.tif
|       |-- MOD09GA_A2001290.sur_refl.tif
|       `-- ...
|-- Datasets
|   |-- AHB
|   |   |-- CSBS
|   |   |   |-- M-0000-0-0.tif
|   |   |   |-- M-2014-4-15.tif
|   |   |   `-- ...
|   |   |-- Landsat
|   |   |   |-- L_0000-0-0.tif
|   |   |   |-- L_2014-2-10.tif
|   |   |   `-- ...
|   |   |-- MODIS
|   |       |-- M_0000-0-0..tif
|   |       |-- M_2014-2-10.tif
|   |       `-- ...
|   |-- Tianjin
|       |-- CSBS
|       |   |-- M-0000-0-0.tif
|       |   |-- M-2013-11-4.tif
|       |   `-- ...
|       |-- Landsat
|       |   |-- L-0000-0-0.tif
|       |   |-- L-2013-11-4.tif
|       |   `-- ...
|       |-- MODIS
|           |-- M-0000-0-0.tif
|           |-- M-2013-11-4.tif
|           `-- ...
|-- LGC
    |-- CSBS
    |   |-- 00000000_CSBS.tif
    |   |-- 20040502_CSBS.tif
    |   `-- ...
    |-- Landsat
    |   |-- 20040416_TM.tif
    |   |-- 20040502_TM.tif
    |   `-- ...
    |-- MODIS
        |-- MOD09GA_A2004107.sur_refl.tif
        |-- MOD09GA_A2004123.sur_refl.tif
        `-- ...
```
The `CSBS` data are generated by Stage-I. Access through: `https://pan.baidu.com/s/1CrgLjPCWsqI5ZmemitlMug?pwd=pqee` code: `pqee`

Make sure to place your dataset files in the appropriate folders as shown above. Note the first time of CSBS data is not needed, but it still needs a placeholder for the code to make time pairs.

## Setting Up the Training Environment

To set up the training environment, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yc-cui/SDCS.git
   cd SDCS/Stage-II
   ```

2. **Create a Virtual Environment:**
   ```bash
   conda env create -f environment.yml
   conda activate STF
   ```

## Running the Training and Testing

To run the training script, use the following command:

```bash
python train.py --dataset CIA --test_freq 1 --device 1 --lr 2e-4
```

### Command Line Arguments

- `--batch_size`: Size of the batches for training (default: 2).
- `--epochs`: Number of epochs to train (default: 1000).
- `--dataset`: Dataset to use (options: "CIA", "LGC", "AHB", "DX", "TJ").
- `--test_freq`: Frequency of testing during training (default: 1).
- `--device`: GPU device index to use (default: 1).
- `--seed`: Random seed for reproducibility (default: 42).
- `--lr`: Learning rate (default: 2e-4).
- `--num_workers`: Number of workers for data loading (default: 8).
- `--pin_mem`: Whether to pin memory (default: True).
- `--wandb`: Use Weights & Biases for logging (default: False).

### Testing the Model

After training, the model will be tested automatically. You can also run a specific test command if needed:

```bash
python train.py --test --dataset CIA --ckpt path/to/your/checkpoint.ckpt
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact
Should you have any question, please contact cugcuiyc@cug.edu.cn
