# PyTorch Project Template

A personal template for projects that use multiple datasets and multiple models. The code is not supposed to be used as a framework but as a starting point with a focus on flexibility.

## Works with
```
pip3 install --upgrade torch==1.3.0
pip3 install --upgrade sacred==0.8.1
```

## Sacred
We use sacred to configure experiments. This allows you to edit the configuration on the command line so you can easily run different experiments.
```
python main.py print_config
python main.py print_config with 'p.dataset_variation="bAbI1k"'
python main.py with 'p.dataset_variation="bAbI1k"'
```

## Features
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
		- copies python files which are found recursively into the log folder
		- optional early stopping mechanism
		- optional maximum number of steps
		- saves last and best model whenever the evaluation loop has finished

While the models and datasets are somewhat general, the trainer is most likely require more project/problem specific changes. The code was written such that this is (hopefully) easy to do.
