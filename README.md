# PeTaL (Periodic Table of Life)

The Periodic Table of Life (PeTaL, pronounced petal) is a design tool aimed at allowing users to seemlesly move from ideas (from nature or other sources) to design.
PeTaL is built around a graph database, machine learning tools, and a website. 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.  

Clone PeTaL:  
```
git clone https://github.com/nasa/PeTaL
cd PeTaL
```  

First, install [poetry](https://python-poetry.org/docs/):  
`curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python`  
Use `poetry` to  install dependencies.  
`poetry install`  
`pytorch` is non-standard in the way it is packaged, and may need to be installed separately: `poetry run pip install torch==1.5.1+cpu torchvision==0.6.1+cpu -f https://download.pytorch.org/whl/torch_stable.html`, however an entry in `pyproject.toml` is still needed.  

**To run the PeTaL pipeline**, use `./run config/default.json`, or choose one of the other config files, such as `airfoil_training.json`, or `mock_ml_config.json` (an integrated test).  
After doing so, the `neo4j` browser can be used to verify that a pipeline has run.  
Also, pipelines must be manually cancelled, as they are designed to run as a server that receives incoming data constantly.  
**To run the PeTaL website**, enter the `site` directory and run `python manage.py runserver`.

Steps to get started if you are running it from scratch:
```
./run config/mock_species_articles.json # Step 1
# Ctrl-C once the database has enough articles for testing
# This creates the ../data/index.html https://github.com/nasa/PeTaL/issues/28#issuecomment-649080792 
./run config/search.json # Step 2 
# Wait until index has been generated in PeTaL/data directory
cd site
python manage.py runserver
```

Note that running the PeTaL pipeline requires an actively running `neo4j` server, with URL and login info entered into a config file.
The default configuration expects a `neo4j` bolt server running on `7687`, with the username "neo4j" and password "life".
*These passwords and URLs are stored in a readable format, so it would be wise not to commit config files containing passwords to a production environment.*

*For specific instructions on extending PeTaL through pipeline modules, read [this documentation](https://github.com/LSaldyt/bitflow/blob/master/README.md)*.
### Guides 
[Setting up the development environment](https://github.com/nasa/PeTaL/wiki/Setting-up-a-development-environment)

## Deployment

Deploying PeTaL is as simple as setting up a `neo4j` server, running the pipeline backend to populate the database, and starting the website, potentially on a separate server.  
Automating this with docker would be desirable.

## Layout

The following section describes the directories of the PeTaL repository.

* #### site

  * The Django code for running the actual PeTaL website, which uses the `neo4j` database created by the pipeline.

* #### modules

  * Modules for PeTaL's data pipeline:

  * ###### mining

    * Holds taxon catalogers, article scrapers, and image downloaders.

  * ###### search

    * Holds multiple indexer modules, which, unsurprisingly, create the search index used by the PeTaL website.

  * ###### taxon\_classifier

    * An image classifier at the taxon level

  * ###### airfoils

    * Experimental machine learning models related to airfoils and wing design

  * ###### directory\_2.0

    * Parsing code for the "Directory 2.0" project, mostly outdated.

  * ###### mock

    * Modules intended for use in integrated testing of the data pipeline.


* #### data

  * Various data for PeTaL's data pipeline. Notably includes the search index and lexicon, images, and all machine learning modules, et cetera.

* #### pipeline

  * Backend code for the data pipeline, but not PeTaL-specific. Separated into another github repository. Some code is overly complicated, so don't feel afraid of contacting Lucas Saldyt if it breaks.

* #### config

  * Configurations for the data pipeline. Specifies a number of modules that should be run and settings related to running them, such as the maximum number of processes.

* #### tests

  * PeTaL's tests. Don't worry. Things will break in production no matter how many of these you write.

* #### docs

  * Documentation. If nothing gets added to this directory, you're doing it right.

## Legacy Code

For Flask version of PeTaL as it existed in 2019, see the /legacy/ directory.
Since the HTML/CSS/Javascript is similar to what is currently used, and some code is shared, this is kept close-by, potentially serving as a reference for future developers to build upon.

One module in the legacy code, Biomole, makes use of code written by Soren Knudsen. The code is available at
[https://github.com/sknudsen/biomole](https://github.com/sknudsen/biomole).

The code makes use of data from AskNature that is not in the published source due to property rights. 
Instead, you can get the data from the links below to see a working version:
 
http://biomole.asknature.org/json/hierarchy.json

http://biomole.asknature.org/json/strategies.json




## Project Team

* **Vik Shyam** - *Principal Investigator*
* **Colleen Unsworth** - *[Product Owner](https://www.mountaingoatsoftware.com/agile/scrum/roles/product-owner)*
* **Herb Schilling** - *Data Science Lead* - Consultant, facilitator, alternate mentor
* **Calvin Robinson** - *Data Architect* - Data architects define how the data will be stored, consumed, integrated and managed by different data entities and IT systems, as well as any applications using or processing that data in some way.
* **Paht Juangphanich** - *Technical Support* - Helps team members overcome any technical challenges they face.
* **Brandon Ruffridge** - *[Scrum Master](https://www.agilealliance.org/glossary/scrum-master/) and Technical Lead*

### PeTaL 3.0

* **Lucas Saldyt** - *Backend development, machine learning* 01-2020 - 04-2020
* **Olakunle Akinpelu** - *Backend development* 01-2020 - 06-2020
* **Kei Kojima** - *   *
* **Elliot Hill** - *   *
* **Benjamin Huynh** - *   *

### PeTaL 2.0 

* **Angeera Naser** - *Front end development and intern team lead Summer 2018*
* **Allie Calvin** - *UI development and data collection* 
* **Bishoy Boktor** - *Backend development* 
* **Brian Whiteaker** - *Text Classification* 
* **Connor Baumler** - *Backend development*
* **Courtney Schiebout** - *Backend development*
* **Isaias Reyes** - *3D visualizations, interactions & full redesign*
* **Jonathan Dowdall** - *Backend development* 
* **Kaylee Gabus** - *Front end development*
* **Kei Kojima** - *Front end development*
* **Lauren Friend** - *Backend development*
* **Manju Johny** - *Backend development*

### PeTaL 1.0 - R

* **Nicholas Bense** - *Backend development*
* **Allie Calvin** - *UI development and data collection* 
* **Victoria Kravets** - *Backend development* 
