# Data Pipeline

PeTaL's Data Pipeline manages sequential data acquisition and processing within PeTaL.
For example, a pipeline process would be obtaining a catalog of living species, using this to obtain articles on species, and then indexing those articles for use in PeTaL's search engine.

## Modularity

Extending the pipeline is meant to be easy.
Modules are simply defined by type signatures on neo4j entry labels, such as `None -> Taxon` for the cataloger or `Taxon -> Article` for article scrapers.
Machine learning modules are commonly defined by something like `Image -> None`, and save their results to file.

See `modules/WikipediaModule.py` for a concrete and well documented example, and also see `modules/module.py` for the base class to derive from.
For an example of an independent module, see `modules/BackboneModule.py`, which populates Species nodes into neo4j.

Within a Module's process() function, self.default\_transaction(data) is used to create a Transaction() object from JSON for node properties. For more advanced data miners, see self.custom\_transaction() and self.query\_transaction() as they are all defined in `modules/module.py`.

