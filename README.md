# EL Authorship Analysis using Python
<img src="https://img.shields.io/badge/-Python-F9DC3E.svg?logo=python&style=flat">
<img src="https://img.shields.io/badge/-Python-F9DC3E.svg?logo=python&style=flat">
<img src="https://img.shields.io/badge/-Python-F9DC3E.svg?logo=python&style=flat">
## Feature
You can use the python scripts and web apps in this repository to perform n-gram searches and analyze author features.

## How to use

Before you run the python script, you need to prepare the dataset you want to analyze.

### Python script

Clone Repository & Run

```
git clone git@github.com:shoji-pg-love/ELPy.git
cd ELPy
python3 Comparison.py -s {step} -m {limit} {dataset_dir} {unknown_txt}

// input
Analyse which n-gram length? (1=uni, 2=bi, 3=tri, etc.  Enter=1-3 all): 


```

![sc](https://github.com/user-attachments/assets/ffd16846-01fe-46e3-a183-ea0b098271d1)

If the analysis is successful, the results should be displayed in the terminal.


### Web App

You select the base file to be analyzed and the comparison file from the file picker on the web application. After setting the conditions of n-gram value, block width and ranking, press the [Analyze] button, and matching n-grams will be highlighted and displayed.

![webapp](https://github.com/user-attachments/assets/3a84627b-76c8-4990-9d7d-d0ffc512aff0)
