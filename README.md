# Hackathon
This repository contains toolkit useful for the partipants of the 2023 PSCC Data challenge

Please document the project the better you can.

# Startup the project

There is multiple way to create a suitable environment for your data challenge
## Virtual env
You can use virtualenv:
Create virtualenv and install the project:
```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv ~/venv ; source ~/venv/bin/activate ;\
virtualenv -p python3.8 myenv
source myenv/bin/activate
```

Install git for later
```bash
pip install python-git
```
or

```bash
conda install -c anaconda git
```
## Conda

You can also create a environment via conda.

### conda installation

[Linux](https://docs.anaconda.com/free/anaconda/install/linux/)
[MacOS](https://docs.anaconda.com/free/anaconda/install/mac-os/)

```bash
conda create -n myenv python=a python version >= 3.8
```

Install git for later
```bash
conda install git
```
Check for hackathon in github.com/{group}. If your project is not set please add it:

## Install

Go to `https://github.com/{group}/hackathon` to see the project, manage issues,
setup you ssh public key, ...

Clone the project and install it:

```bash
git clone git@github.com:{group}/hackathon.git
cd hackathon
pip install -r requirements.txt
make clean install test                # install and test
```
## Usage
Once you have made your predictions, you will have to save them as NIfTI files (.gz or not). Then you can create a .csv submission file
as follow:
```python
from hackathon.submission_gen import submission_gen

predictionpath = 'path'
csvpath = 'csvpath'
submission_gen(predictionpath, csvpath)
"submission file saved at 'csvath"
```
