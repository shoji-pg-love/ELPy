# EL Authorship Analysis using Python
<img src="https://img.shields.io/badge/-Python-F9DC3E.svg?logo=python&style=flat">
<img src="https://img.shields.io/badge/-Python-F9DC3E.svg?logo=python&style=flat"> 

[![GitHub Pages](https://img.shields.io/static/v1?label=GitHub+Pages&message=+&color=brightgreen&logo=github)](https://shoji-pg-love.github.io/ELPy/) 

## Feature
You can use the python scripts and web apps in this repository to perform n-gram searches and analyze author features.

[N-Gram Comparer Web APP](https://shoji-pg-love.github.io/ELPy/)

## How to use

Before you run the python script, you need to prepare the dataset you want to analyze.

### Python script

Clone Repository & Run

```
$ git clone git@github.com:shoji-pg-love/ELPy.git
$ cd ELPy
$ python3 Comparison.py -s {step} -m {limit} {dataset_dir} {unknown_txt}

// input
Analyse which n-gram length? (1=uni, 2=bi, 3=tri, etc.  Enter=1-3 all): 


```

![sc](https://github.com/user-attachments/assets/ffd16846-01fe-46e3-a183-ea0b098271d1)

If the analysis is successful, the results should be displayed in the terminal.


### Web App

You select the base file to be analyzed and the comparison file from the file picker on the web application. After setting the conditions of n-gram value, block width and ranking, press the [Analyze] button, and matching n-grams will be highlighted and displayed.

![webapp](https://github.com/user-attachments/assets/3a84627b-76c8-4990-9d7d-d0ffc512aff0)


## how to use KWICs
#### 1_1
(e.g.) python kwic.py <filename> <target>
```python KWIC1_1.py sample.txt "climate change"```

#### 1_2
(e.g.)python KWIC1_2.py <filename> <type> <target>
```
python KWIC1_2.py sample.txt token "climate"
python KWIC1_2.py sample.txt pos "ADJ NOUN"
python KWIC1_2.py sample.txt ent "PERSON"
```

#### 2_1
same as 1_2

  
# note
We used generative AI tools to develop this app.