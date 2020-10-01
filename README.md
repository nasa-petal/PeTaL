# PeTaL (Periodic Table of Life)

The Periodic Table of Life (PeTaL, pronounced petal) is a design tool aimed at allowing users to seemlesly move from ideas (from nature or other sources) to design.

PeTaL is comprised of multiple interconnected services. This repository is for the ReactJS web front end client. There are other repositories for the [API](https:/www.github.com/nasa/petal-api) and [Labeller](https:/www.github.com/nasa/petal-labeller).

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project to a production system.  

Uses the [create-react-app material-ui example](https://github.com/mui-org/material-ui/tree/master/examples/create-react-app).

After cloning this repo run:    
`npm install`    
`npm start`    

or    
`yarn install`    
`yarn start`

## Contributing

We're using the [Git Feature Branch workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/feature-branch-workflow), to contribute code to the code base. To summarize, `master` should always contain only stable, tested code. To work on a change, first create a feature branch off of `master`, perform your changes, and when your code is ready, submit a pull request to merge your branch with `master`. 

## Deployment

`npm run deploy`    
https://create-react-app.dev/docs/deployment/#github-pages    

View site at https://nasa.github.io/PeTaL/

## Layout

## Legacy Code

For the Django version as it existed in 2020 see the master-legacy-2020 branch.
For Flask version of PeTaL as it existed in 2019, see the /legacy/ directory in the master-legacy-2020 branch.

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
* **Alex Ralevski** - *[Product Owner](https://www.mountaingoatsoftware.com/agile/scrum/roles/product-owner)*
* **Herb Schilling** - *Data Science Lead* - Consultant, facilitator, alternate mentor
* **Calvin Robinson** - *Data Architect* - Defines how the data will be stored, consumed, integrated and managed by different data entities and IT systems, as well as any applications using or processing that data in some way.
* **Paht Juangphanich** - *Technical Support* - Helps team members overcome any technical challenges they face.
* **Brandon Ruffridge** - *[Scrum Master](https://www.agilealliance.org/glossary/scrum-master/) and Technical Lead*

### PeTaL 0.4

* **Shruti Janardhanan** - *   * 08-2020 -
* **Jerry Qian** - 08-2020 -

### PeTaL 0.3

* **Lucas Saldyt** - *Backend development, machine learning* 01-2020 - 04-2020
* **Olakunle Akinpelu** - *Backend development* 01-2020 - 06-2020
* **Kei Kojima** - *   *
* **Elliot Hill** - *   *
* **Benjamin Huynh** - *   *

### PeTaL 0.2 

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

### PeTaL 0.1 - R

* **Nicholas Bense** - *Backend development*
* **Allie Calvin** - *UI development and data collection* 
* **Victoria Kravets** - *Backend development* 
