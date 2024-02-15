# MolPropPred


### Clone

This repo uses git submodules for the MolPropPredTool. 
Therefore, you need to clone the repository with `--recurse-submodules`.

```
git clone --recurse-submodules https://github.com/Shreyas-Bhat/MolPropPred
```

In case you forget to do that, you need to run the following command:

```
git submodule update --init
```
 
The tool can be accessed in `MolPropPredTool` folder if cloned using git or downloaded zip from the [repo](https://github.com/MrunmayS/MolPropPredTool)
## QMolPropPredTool V0.1


### Prerequisites

* Python 3.10
* Python Dependencies :
	- numpy
	- deepchem
	- matplotlib==3.3.4
	- pandas==1.1.5
	- rdkit_pypi
	- scikit_learn
	- torch
	- tensorflow

### Installation :

It is recommended to use a Virtual Environment to install the dependencies. You may require administrator privileges to install the Virtual Environment. Installation instructions have been tested on Ubuntu, but should also work on
other Linux based operating systems and MacOS. 

1.	Install virtualenv with `pip install virtualenv`. 
2.	Create the virtual environment with `virtualenv qm9`. 
3.	Activate the virtual environment with `source qm9/bin/activate`. 
4.	Install the dependencies in the virtual environment with 	`pip install -r requirements.txt`.
5.	Type `deactivate` to deactivate the 	Virutal Environment once you are done.

or run the following

```bash
pip install virtualenv
virtualenv qm9
source qm9/bin/activate
pip install -r requirements.txt
```
-------------------------------------------------------------------------------

## QMolPropPredTool Usage :

```bash
python3 main.py <SMILE-string>
```


