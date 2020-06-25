
import os
import json
from .utils import get_module_names, fetch

def create_dependencies(directory='modules'):
    '''
    Read bitflow modules in a directory and get their type signatures
    Dump these to a JSON file.

    **Potentially uses a lot of memory, and is separated for this reason**
    '''
    print('CREATING DEPENDENCIES', flush=True)
    dependencies = dict()
    for name in get_module_names(directory=directory):  
        try:
            module = fetch(name, directory=directory)
            print('  Recognized module: ', name, flush=True)
            in_type  = module.in_label
            out_type = module.out_label
            page_batches = module.page_batches
            dependencies[name] = (in_type, out_type, page_batches)
        except:
            print("Unrecognized module: " + name)
        
    with open('.dependencies.json', 'w') as outfile:
        json.dump(dependencies, outfile, indent=4)

if __name__ == '__main__':
    create_dependencies()
