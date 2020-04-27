# Machine Learning in PeTaL

The PeTaL data pipeline separates modules into those used for machine learning, and those used for mining data directly.
However, both modules implement the same interface, and are only separated for organization.
The interface is simply a type signature (in terms of a `neo4j` label), and a `process()` function that operates on that label.
Additionally, the machine learning modules commonly inherit from an `OnlineLearner` class, which defines common operations for online machine learning, for instance a save file where the model is dumped to periodically. Alternatively, the `OnlineTorchLearner` class can be used for pytorch-specific modules.

Typically, a machine learning module will only define an input, for instance `Image -> None` as a type signature.
Of course, it is possible to define any type signature.

## Detailed Example

See `AirfoilRegressor` for detailed documentation. 
This class receives Airfoils from the UIUC database, which are mined by the `Airfoils` mining module, which has type signature `None -> Airfoil`.
The `AirfoilRegressor` currently implemented has a type signature of `Airfoil -> None`, but could easily be changed to `Airfoil -> Metrics` if the predicted airfoil metrics were to be kept in the `neo4j` database.
Additionally, another machine learning module could potentially utilize the model that is trained by `AirfoilRegressor`, in particular on new airfoils, especially those from biology.

Similarly, there are also `AirfoilCreator` and `AirfoilRegimeRegressor` modules. 
`AirfoilCreator` takes desired metrics as input, and gives an airfoil geometry as output.
`AirfoilRegimeRegressor` takes geometry and metrics, and then guesses the regime that the airfoil was tested at.

