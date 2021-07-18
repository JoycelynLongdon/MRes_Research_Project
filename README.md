# MRes Project - The Feasibility of Deriving Forest Change Drivers in DRC with Machine Learning

 [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
 <a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

## Introduction

This repository contains the code for the MRes Research Project for the AI4ER CDT, to investigate the feasibility of applying classic, establishedsupervised machine learning methods to a specific and important environmental policy question. Iinvestigate the ability of these techniques to build on current forest change detection methods andprovide more detailed insights for policy analysis and implementation in the study region. There search questions guiding this enquiry were

## Materials and Method

In this work, a Random Forest Classifier and a Support Vector Machine, trained with the [European space Agency Climate Change Initiative Landcover data set (ESA CCI)](http://www.esa-landcover-cci.org/?q=node/164), were applied to remotely sensed imagery of the PIREDD Plateau REDD Project (within the Mai Ndombe province), acquired from [Google Earth Engine](https://developers.google.com/earth-engine/guides/playground#:~:text=The%20Earth%20Engine%20(EE)%20Code,JavaScript%20code%20editor). Year to year comparison of the resulting Landcover classification maps allowed for thedetection of Landcover conversion events and were compared to established change detection techniques.

## Requirements
- Python 3.8+
- See requirements directory for the full list of dependencies

To instal requirements:
```
pip install -r requirements/requirements.txt
```


## Project Organization
```
├── LICENSE
├── Makefile           <- Makefile with commands like `make init` or `make lint-requirements`
├── README.md          <- The top-level README for developers using this project.
|
├── notebooks          <- Jupyter notebooks
│   └── exploratory    <- Notebooks for initial exploration and experimentation.
|
├── requirements       <- Directory containing the requirement files.
│
├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── preprocessing  <- Scripts to process raw satellite imagery from GEE and for further processing in Python
|   |
│   ├── models         <- Model code for RF and SVM
│   │
│   └── analysis       <- Scripts for data post-processing to retrieve statistics and change maps
│
└── setup.cfg          <- setup configuration file for linting rules
```

---

Project template created by the [Cambridge AI4ER Cookiecutter](https://github.com/ai4er-cdt/ai4er-cookiecutter).
