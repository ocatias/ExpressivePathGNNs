# The Expressive Power of Path based Graph Neural Networks

Code repository for our paper _"The Expressive Power of Path based Graph Neural Networks"_ (ICML 2024). This repository contains only parts of all experiments, the code for EXP and SR can be found [here](https://github.com/tamaramagdr/synthetic-pain).

## Setup
Clone this repository and open the directory

Add this directory to the python path. Let `$PATH` be the path to where this repository is stored (i.e. the result of running `pwd`).
```
export PYTHONPATH=$PYTHONPATH:$PATH
```

Create a conda environment (this assume miniconda is installed)
```
conda create --name pathGNN
```

Activate environment
```
conda activate pathGNN
```

Install dependencies
```
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 -c pytorch
conda install -c pyg pyg=2.2.0
python -m pip install -r requirements.txt
```

### Tracking
Training and different experiments are tracked via [wandb](https://wandb.ai/). If you want to make use of the tracking you need a wandb account. The first time you train a model, you will be prompted to enter you wandb API key. If you want to disable tracking you can do this in the config `Configs/config.yaml`.

## Rerunning Experiments
You can recreate the experiments with the following commands. The results will be stored in the `Results` folder.

- PAIN on CSL:
```
bash Scripts/run_csl_exp.sh
```

- PAIN on ZINC (this will also give you our runtime evaluation):
```
python Exp/run_experiment.py -dataset ZINC -grid Configs/Benchmark/ZINC_pathGNN.yaml --repeats 10
```

- PAIN on MOLHIV:
```
python Exp/run_experiment.py -dataset ogbg-molhiv -grid Configs/Benchmark/molhiv_pathGNN_V3.yaml --repeats 10
```

- PAIN on EXP & SR: Please refer to [this repository](https://github.com/tamaramagdr/synthetic-pain).

- Runtime benchmark of GIN, DS and DSS on ZINC: Please refer to [this repository](https://github.com/ocatias/GNN-Simulation).


## Citation

If you use our code please cite us as
```
@inproceedings{pathGNNs2024,
title={The Expressive Power of Path based Graph Neural Networks},
author={Drucks,Tamara and Graziani, Caterina and Jogl, Fabian and Bianchini, Monica and  Scarselli, Franco and Gärtner, Thomas },
booktitle={ICML},
year={2024},
url={https://openreview.net/forum?id=io1XSRtcO8}
}
```
