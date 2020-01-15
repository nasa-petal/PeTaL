# PeTaL (Periodic Table of Life)

The Periodic Table of Life (PeTaL, pronounced petal) is intended to be a design tool to enable a deeper understanding of natural systems and to enable the creation or improvement of nature-inspired systems. The tool includes an unstructured database, data analytics tools and a web-based user interface. Three levels of information are expected to be captured: morphology that would aid designers by providing context-specific physical design rules, function-morphology relationships to provide nature-inspired solution strategies, and system-level relationships that involve the interaction of several biological models including flow of resources and energy. In its current form, PeTaL is structured as a large NoSQL database that will be accessible to researchers and citizen scientists. It includes entomological and paleontological data from the Cleveland Museum of Natural History (CMNH) in Cleveland, OH, the Cincinnati Museum Center in Cincinnati, OH and the Smithsonian. PeTaL can display relationships between biological models, geography, and environment through maps and plots. These may be used to glean patterns or design rules. Data can also be downloaded for further analysis. A more systematic design process is under development that will allow multiple models to be used for the various stages of design.

The current test case for PeTaL – the idea we need to build a working demo for – is a bio-inspired thermal management/distribution system.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

First fork the project from gitlab and clone it into your local machine with command:

```
git clone git@gitlab.grc.nasa.gov:[Your user name]/petal.git
```

You will need to take care of SSH keys to be able to pull and push
Follow instructions on gitlab for this


## Run manually (Recommended)

### Prerequisites

Anaconda
Python3

To install Anaconda:

[install from anaconda.com](https://www.anaconda.com/download/)

### Installing

Create Anaconda enviroment:
```
conda create -n petal_env  
```

Activating Anaconda enviroment:
cd into petal directory and type
Mac:
```
source activate petal_env 
```
Windows:
```
conda activate petal_env
```

Get required dependencies
Mac:
```
conda install --file req_conda.txt
pip3 install -r requirements.txt
python3 -m spacy download en
```
Windows:
```
conda install --file req_conda.txt
pip install -r req_pip.txt
python -m spacy download en
```
Alternate method used before Biocene
```
conda create -n petal_env python=3.7
pip install pandas flask networkx keras numpy pillow sqlalchemy Flask-SQLAlchemy scikit-learn tensorflow
pip install ontospy wikipedia pyLDAvis gunicorn rdflib spaCy
pip install nltk
python -m spacy download en
```

### Running

Run the application from command line in project root directory 
Mac:
```
python3 run.py 
```
Windows:
```
python run.py
```
Alternatively you can specify data paths
```
python3 run.py [--ont=path/to/OWL/file --data=path/to/data/file --lit=path/to/literature/csv]
```

Data paths default to: 

```
ont=data/ontology.csv
data=data/classes.csv
lit=data/lit.csv
```

## Run with docker

Configure your docker settings first.

Check on: General -> Expose daemon on tcp://localhost:2375 without TLS 

Add: Daemon -> Insecure registrites: sciapps.grc.nasa.gov:5000

Once docker is running, enter this command on a command line from the project directory:

```
docker build . -t petal_img && docker run --name petal --rm -it -p 5000:5000 petal_img
```

You can then access petal from localhost:5000 in your browser.


## CSS

Updates to style should be done in .scss files located in /petal/static/sass.
These files are imported in main styles.scss file and all compiled into styles.css
To get your environment ready first install compass following this tutorial:
[compass install](http://thesassway.com/beginner/getting-started-with-sass-and-compass)

After installation you should be able to simply navigate to root petal directory 
from terminal or command line and type:
```
compass watch
```
Your changes to .scss files will then be automatically compiled into styles.css file everytime you save
a .scss file.

## Built With

* [flask](http://flask.pocoo.org) - The web framework used
* [pandas](https://pandas.pydata.org/) - Data Analysis library
* [network x](https://github.com/networkx/networkx) - Graph building library
* [Ontospy](https://github.com/lambdamusic/OntoSpy/wiki) - python library for interfacing with OWL file
* [rdflib](https://github.com/RDFLib/rdflib) - Used to translate OWL files into N-Triples
* [keras](https://keras.io/) - Machine Learning library used for image classification
* [d3.js](https://d3js.org/) - JavaScript library for visualizations

## Authors

* **Vik Shyam** - *Project Lead*

* **Angeera Naser** - *Team Lead and frontend development*
* **Allie Calvin** - *UI development and data collection* 
* **Jonathan Dowdall** - *Backend development* 
* **Bishoy Boktor** - *Backend development* 
* **Brian Whiteaker** - *Text Classification* 
* **Isaias Reyes** - *3D visualizations, interactions & full redesign*
* **Kaylee Gabus** - *Front end development*
* **Lauren Friend** - *Backend development*
* **Connor Baumler** - *Backend development*
* **Courtney Schiebout** - *Backend development*
* **Manju Johny** - *Backend development*
* **Kei Kojima** - *Front end development*

