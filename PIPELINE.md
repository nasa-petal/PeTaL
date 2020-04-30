# Data Pipeline

PeTaL's data pipeline is responsible for building the `neo4j` database used by the main PeTaL website.
It contains web scrapers and machine learning tools, which are chained together by defining type signatures on `neo4j` node labels.
For example, the species catalog module generates `Taxon` nodes, and the wikipedia article module receives `Taxon` nodes and creates `WikipediaArticle` nodes.

## Extending Mining Capability

The pipeline can be extending by creating a module, for instance in the `modules/mining/` directory.  
A module is defined by a type signature (type in) -> (type out, label from, label to) and a process(node : type in) function which creates a list of Transaction() objects from (type in) nodes.  
Here, "types" are neo4j labels. 

#### Independent
A basic skeleton of an "independent" module (with no inputs) looks like this:
```python
from pipeline.utils.module import Module

class MyModule(Module):
    def __init__(self, in_label=None, out_label='Output', connect_labels=None, name='MyModule'):
        Module.__init__(self, in_label, out_label, connect_labels, name)

    def process(self):
    	for json_data in ...:
		yield self.default_transaction(json_data) # Create new nodes of type 'Output'
```
A good example of this is `modules/mining/OptimizedCatalog.py`

#### Dependent
A basic skeleton of a "dependent" module looks like this:
```python
from pipeline.utils.module import Module

class MyModule(Module):
    def __init__(self, in_label='Input', out_label='Output', connect_labels=('to', 'from'), name='MyModule'):
        Module.__init__(self, in_label, out_label, connect_labels, name)

    def process(self, previous):
        data = previous.data # Get the neo4j JSON of a node with label 'Input'
	# new_data = ...
	yield self.default_transaction(new_data)
```
A good example of this is `modules/mining/WikipediaModule.py`

Within a Module's process() function, self.default\_transaction(data) is used to create a Transaction() object from JSON for node properties. For more advanced data miners, see self.custom\_transaction() and self.query\_transaction() as they are all defined in `modules/mining/module.py`.

#### Machine Learning

Relevant base classes to machine learning live in `pipeline/utils`.  
In particular, `BatchLearner`, `BatchTorchLearner`, `OnlineLearner`, and `OnlineTorchLearner` are worth looking at.

A basic skeleton of a neural-network based machine learning module in PeTaL looks like this:
```python
from petal.pipeline.utils.BatchTorchLearner import BatchTorchLearner

class MyMLModule(BatchTorchLearner):
    def __init__(self, filename='data/models/my_ML_module.nn'):
    	# Change these based on the underlying ML model, see BatchTorchLearner documentation.
        BatchTorchLearner.__init__(self, nn.CrossEntropyLoss, optim.SGD, dict(lr=0.001, momentum=0.9), in_label='Input', name='MyMLModule', filename=filename)

    def init_model(self):
        self.model = TorchModel(..)

    def transform(self, node):
    	# Process node.data into inputs and outputs
        yield inputs, outputs
```
See `modules/taxon_classifier/TaxonClassifier` for an example of this.

A more advanced neural network example might look like this.
Both examples use the same base class, but more fine-grained control is given by overloading more functions.
```python
class MyMLModule(BatchTorchLearner):
    def __init__(self, filename='data/models/my_model.nn', name='MyMLModule'):
        BatchTorchLearner.__init__(self, filename=filename, epochs=2, train_fraction=0.8, test_fraction=0.2, validate_fraction=0.00, criterion=nn.MSELoss, optimizer=optim.Adadelta, optimizer_kwargs=dict(lr=1.0, rho=0.9, eps=1e-06, weight_decay=0), in_label='Input', name=name)

    def init_model(self):
        self.model = TorchModel(..)

    # def learn() inherited, uses transform()
    def transform(self, node):
	yield inputs, outputs

    def test(self, batch):
    	# Process a test batch (given 20% of the time, based on test_fraction parameter above)

    def val(self, batch):
    	# Process a validation batch (given 20% of the time, based on test_fraction parameter above)
```

## Scheduler, Driver, and Pipeline classes

Behind the scenes, this is how the pipeline works at a very high level. 
This code is (if I may say so) well documented, because I saw it as being the hardest to understand or fix.
See the top-level of the `pipeline` directory.

#### Scheduler

Scheduler will load any modules importable from the `modules` subdirectories.
It expects a file containing a class of the same name.
For example `modules/mymodules/MyModule.py` with `class MyModule: ...` within the file is a valid setup.
Also, each module should derive from a base `Module` class (or another class that derives from `Module`.
As documented above, these are located in `pipeline.utils`.

Scheduler reads the type signatures of modules, and runs them based on this.  
For instance, `OptimizedCatalog` is "indepdent", because it generates `Taxon` nodes without any input, so this is run initially.  
Then, once Species nodes are created, modules which rely on them are scheduled and eventually run, with respect to the amount of nodes available.  
For instance, `WikipediaModule`, `EOLModule`, and `JEBModule` will all run after BackboneModule has generated 10 nodes.

#### Driver

Driver is just a connection to a neo4j database.
Essentially it enables some useful abstraction over the `neo4j` api, specifically allowing the developer to worry only about the JSON containing in nodes, and their labels and connections.
This is done by using the `Transaction` class, located in `pipeline/utils`.
For further understanding, see the file-level documentation.

#### Pipeline

Pipeline is an interface which allows the server to dynamically load modules and settings (like how a Djano site supports changing files while the website is running).
It's really that simple, but it's also documented at the file-level in the `pipeline` folder.

