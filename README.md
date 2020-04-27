# PeTaL (Periodic Table of Life)

The Periodic Table of Life (PeTaL, pronounced petal) is intended to be a design tool to enable a deeper understanding of natural systems and to enable the creation or improvement of nature-inspired systems. The tool includes an unstructured database, data analytics tools and a web-based user interface. Three levels of information are expected to be captured: morphology that would aid designers by providing context-specific physical design rules, function-morphology relationships to provide nature-inspired solution strategies, and system-level relationships that involve the interaction of several biological models including flow of resources and energy. In its current form, PeTaL is structured as a large `neo4j` database, accessible to researchers and citizen scientists. It includes data on every living species.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

## Layout

The following section describes the directories of the PeTaL repository.

#### site

The Django code for running the actual PeTaL website, which uses the `neo4j` database created by the pipeline.

#### modules

Modules for PeTaL's data pipeline:

###### mining

Holds taxon catalogers, article scrapers, and image downloaders.

###### search

Holds multiple indexer modules, which, unsurprisingly, create the search index used by the PeTaL website.

###### taxon\_classifier

An image classifier at the taxon level

###### airfoils

Experimental machine learning models related to airfoils and wing design

###### directory\_2.0

Parsing code for the "Directory 2.0" project, mostly outdated.

###### mock

Modules intended for use in integrated testing of the data pipeline.


#### data

Various data for PeTaL's data pipeline. Notably includes the search index and lexicon, images, and all machine learning modules, et cetera.

#### pipeline

Backend code for the data pipeline, but not PeTaL-specific. Separated into another github repository. Some code is overly complicated, so don't feel afraid of contacting Lucas Saldyt if it breaks.

#### config

Configurations for the data pipeline. Specifies a number of modules that should be run and settings related to running them, such as the maximum number of processes.

#### tests

PeTaL's tests. Don't worry. Things will break in production no matter how many of these you write.

#### docs

Documentation. If nothing gets added to this directory, you're doing it right.

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

