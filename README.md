# PyTorch Project Template

A personal template for projects that use multiple datasets and multiple models. The code is not supposed to be used as a framework but as a starting point with a focus on flexibility.

## Works with
```
pip3 install --upgrade torch==1.3.0
pip3 install --upgrade sacred==0.8.1
pip3 install --upgrade munch==2.5.0
pip3 install --upgrade tensorboardX==1.9.0
pip3 install --upgrade numpy=1.17.2
```

## Usage
We use sacred to configure experiments. This allows you to edit the configuration on the command line so you can easily run different experiments.
```
python main.py print_config
python main.py print_config with p.hidden_size=512
python main.py with p.dataset_variation=bAbI1k
python main.py with p.dataset_variation=bAbI1k p.hidden_size=512
```

## Features (for now)
This code base comes with the following features:
- modular datasets
	- bAbI v1.2
- modular models
	- simple sequence classification lstm
- modular trainers
	- basic_trainer
		- manages train and evaluation loop
		- can log and evaluate at individual number of steps
		- creates a log folder and warns if it exists
		- stores logs in text file
		- stores tensorflow event files
		- copies python files which are found recursively into the log folder
		- optional early stopping mechanism
		- optional maximum number of steps
		- saves last and best model whenever the evaluation loop has finished
		- can continue where a previous experiment has stopped

While the models and datasets are somewhat general, the trainer is most likely going to require more project/problem specific changes. The code was written such that this is (hopefully) easy to do.

