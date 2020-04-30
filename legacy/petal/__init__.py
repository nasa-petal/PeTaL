from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import petal.graph
import sys
import os
import petal.vision
import argparse


# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--ont', default='petal/data/ontology.owl')
parser.add_argument('--data', default='petal/data/classes.csv')
parser.add_argument('--lit', default='petal/data/literature.csv')
parser.add_argument('--debug', default=True)
args = vars(parser.parse_args())

ONT_PATH = args['ont']
DATA_PATH = args['data']
LIT_PATH = args['lit']
DEBUG = args['debug']

ONT_PATH = 'petal/data/ontology.owl'
DATA_PATH = 'petal/data/classes.csv'
LIT_PATH = 'petal/data/literature.csv'
DEBUG = True

# Build classifier model
global resnet
resnet = vision.ResNet()
global geonet
geonet = vision.GeoNet()

# Build graph from source data
print("Building graph...")
ont = graph.build_graph(ONT_PATH, DATA_PATH, LIT_PATH)

# Start flask app
print("Starting Flask...")
app = Flask(__name__)

# Database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db' # ...///... for relative path

# New db instance
db = SQLAlchemy(app)

from petal import routes