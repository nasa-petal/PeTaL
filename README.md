# PeTaL (Periodic Table of Life)

The Periodic Table of Life (PeTaL, pronounced petal) is a design tool aimed at allowing users to seemlesly move from ideas (from nature or other sources) to design.
PeTaL is built around a graph database, machine learning tools, and a website. 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

```
git clone https://github.com/nasa/PeTaL
cd PeTaL
pip install -r requirements.txt
```

**To run the PeTaL website**, enter the `site` directory and run `python manage.py runserver`.

**To run the PeTaL pipeline**, use `./run config/default.json`, or choose one of the other config files, such as `airfoil_training.json`, or `mock_ml_config.json` (an integrated test).

Note that running the PeTaL pipeline requires an actively running `neo4j` server, with URL and login info entered into a config file.
The default configuration expects a `neo4j` bolt server running on `7687`, with the username "neo4j" and password "life".
*These passwords and URLs are stored in a readable format, so it would be wise not to commit config files containing passwords to a production environment.*

*For specific instructions on extending PeTaL through pipeline modules, read [this documentation](https://github.com/LSaldyt/bitflow/blob/master/README.md)*.

## Deployment

Deploying PeTaL is as simple as setting up a `neo4j` server, running the pipeline backend to populate the database, and starting the website, potentially on a separate server.

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
Since the HTML/CSS/Javascript is similar to what is currently used, and some code is shared, this is kept close-by, potentially serving as a reference for future interns to build upon.

## Authors

* **Vik Shyam** - *Principal Investigator*
* **Paht Juangphanich** - *Technical Lead*

## PeTaL 1.0 - R

* **Nicholas Bense** - *Backend development*
* **Allie Calvin** - *UI development and data collection* 
* **Victoria Kravets** - *Backend development* 

## PeTaL 2.0

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

## PeTaL 3.0

* **Lucas Saldyt** - *Backend development, machine learning*
* **Olakunle Akinpelu** - *Backend development*


