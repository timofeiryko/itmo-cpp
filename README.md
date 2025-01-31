# ITMO CPP

All the code related to our CPPs paper in SCAMT (ITMO). Tasks are managed in the related project on GitHub.

## Workflow & good practices

1) Pull this repo before you start some work and push to it, when you have done something

2) Create a separate branch, when you start working on some part of the project ("feature"). Commit every piece of your work. Create pull requests, when you complete the feature

3) Attach meaningful messages to your commits, give python modules, notebooks, funtions and variables clear names

4) Code lives in python modules as reusable functions (make sure that these functions can be used in different contexts: e. g., don't hardcode the column names, make it the parameters of the function). It's OK to prototype in the notebook, but if some part of the code is going to be used again, especially by other people, put it into module

5) Vizualizations and your structured comments live in noteboooks

6) Before uploading the notebook, please, restart it and run all the cells to make sure, that all is ok. Separate long steps (e.g. training or fine-tuning) to the python modules and don't execute these steps in the notebooks. Instead, load the weights or pickled objects

## Data

- Most of the data used in this project is from the [POSEIDON dataset](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-024-00810-7)

- We create our own relational database in the kaggle notebook. Output files of this notebook are in the `input_data` here. You can check out the [entity relationship diagram](https://editor.ponyorm.com/user/timofeiryko/CPP/designer)

## Project structure

- Each notebook is a `main.ipunb` file in separate folder with `input` and `output` subfolders

- Important output data, generated in the scripts and notebooks, used across multiple parts of the project is in `output_data` folder