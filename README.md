# Local Setup Guide for Model
This repository contains the code for setting up the model

### Setting Up the Virtual Environment

1. Begin by installing the virtual environment package by running:

```
pip install virtualenv
```
2. Create a Python virtual environment at the root directory of the project:

``` 
python<version> -m venv <virtual-environment-name> (e.g python3.10 -m venv env) 
```

3. Activate the virtual environment:

```
source <virtual-environment-name>/bin/activate
```

4. Install required libraries:

```
pip install -r requirements.txt
```

### Run training (Examples)

```bash
python train.py   

python train.py optimizer.lr=0.0003 experiment.train.batch_size=128

python train.py experiment=debug
```


