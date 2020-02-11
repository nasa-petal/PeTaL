# Data Mining

## Extending Mining Capability

This module defines data mining functionality for PeTaL.
Essentially, this functionality is set up by creating modules in the `Module` folder.
A module is defined by a type signature (type in) -> (type out, label from, label to) and a process(node : type in) function which creates a list of Transaction() objects from (type in) nodes.
Here, "types" are neo4j labels. See `modules/WikipediaModule.py` for a concrete and well documented example, and also see `modules/module.py` for the base class to derive from.
For an example of an independent module, see `modules/BackboneModule.py`, which populates Species nodes into neo4j.

Within a Module's process() function, self.default\_transaction(data) is used to create a Transaction() object from JSON for node properties. For more advanced data miners, see self.custom\_transaction() and self.query\_transaction() as they are all defined in `modules/module.py`.

## Scheduler, Driver, and Pipeline classes

#### Scheduler

This library relies on two core things: The scheduler, and the modules passed to it. Scheduler will load any modules importable from the modules subdir (so `__init__.py` *needs* to be edited to add a module), with respect to the blacklist defined in `settings.json`. Scheduler reads the type signatures of modules, and runs them based on this. For instance, `BackboneModule` is "indepdent", because it generates Species nodes without any input, so this is run initially. Then, once Species nodes are created, modules which rely on them are scheduled and eventually run, with respect to the amount of nodes available. For instance, `WikipediaModule`, `EOLModule`, and `JEBModule` will all run after BackboneModule has generated 1000 nodes.

#### Driver

Driver is just a connection to a neo4j database. It has two important methods: run() and run\_page() for "indepedent" and "dependent" modules separately.

#### Pipeline

Pipeline is an interface which allows the server to dynamically load modules and settings. That's it. Really.
