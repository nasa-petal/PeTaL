# PeTaL (Periodic Table of Life)

The Periodic Table of Life (PeTaL, pronounced petal) is a design tool aimed at allowing engineers to find inspiration in nature.
PeTaL is build around a graph database, machine learning tools, and a website. 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

Since PeTaL separates the data pipeline as a submodule, PeTal should be downloaded with: `git clone --recursive-submodules https://github.com/nasa/PeTaL`.
If PeTaL was already downloaded, `git submodule update --init --recursive` updates each submodule.
The pipeline repository currently lives [here](https://github.com/LSaldyt/PeTaL-pipeline).
Modifying it hopefully isn't necessary for most tasks.

To run the PeTaL website, enter the `site` directory and run `pip install -r requirements.txt` and then `python manage.py runserver`.
This is a standard configuration of a Django site.

To run the PeTaL pipeline, enter the `pipeline` directory and run `pip install -r requirements.txt`.
Then, go back to the top-level PeTaL directory and call the `./run` script with a configuration file.
For instance, use `./run config/default.json` to populate a `neo4j` database with species, articles, and images, or `./run config/airfoil_training.json` to train airfoil-related machine learning modules.

Note that running the PeTaL pipeline requires an actively running `neo4j` server, with defaults inserted into each config file.
Also note that these passwords and URLs are stored in a readable format, so it would be wise not to commit config files containing passwords to a production environment.

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

For Flask version of PeTaL as it existed in 2019, see the /legacy/ directory

## Authors

* **Vik Shyam** - *Project Lead*
* **Paht Juangphanich** - *Project Supervisor*

* **Angeera Naser** - *Team Lead and frontend development*
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
* **Lucas Saldyt** - *Backend development, machine learning*
* **Manju Johny** - *Backend development*
* **Olakunle Akinpelu** - *Backend development*

