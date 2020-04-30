def read_input(inpt):
    '''
    Takes a string input and returns the input casts it to the appropriate primitive type

    Parameters
    ----------
    inpt : `str`
        The input to be cast

    Returns
    -------
    `NoneType`, `int`, `float`, `bool`, or `str`
        The cast input
    '''

    if inpt == "None":
        return None

    try:
        return int(inpt) # int is the most constrained type
    except ValueError:
        try:
            return float(inpt)
        except ValueError:
            if inpt == "True":
                return True
            elif inpt == "False":
                return False
            else:
                # Didn't match any other primitive type. Return as string.
                return inpt

def get_types_from_list(l):
    '''
    Makes a set of all types found in a list other than str and NoneType

    Parameters
    ----------
    l : `list` [`any`]
        List of object we want the types from

    Returns
    -------
    types: `set` [`str`]
        The set of types in the list not including NoneType or str
    '''
    types = set([])
    for item in l:
        if type(item).__name__!="str" and type(item).__name__!="NoneType":
            types.add(type(item))
    return types


def get_hyperparam_selections(hyperparam_df):
    '''
    Asks for new values for and updates DataFrame of model hyperparameters

    Parameters
    ----------
    hyperparam_df : `DataFrame`
        DataFrame containing the parameters, their default values, their current values, and their info strings

    Returns
    -------
    hyperparam_df : `DataFrame`
        The DataFrame with updated current values
    '''

    param = ""
    while True:
        print("\nParameters available to change are:\n",", ".join(hyperparam_df['param'].tolist()))
        param = input("\nEnter a parameter name you'd like to change or press enter to quit: ")
        if not param:
            break
        elif param not in hyperparam_df['param'].tolist():
            print("Unrecognized parameter.")
        else:
            cur_index = hyperparam_df.loc[hyperparam_df['param'] == param].index[0]

            cur = hyperparam_df.at[cur_index, 'current_value']

            options = hyperparam_df.at[cur_index, 'availible_values']
            
            list_type = None
            type_list = None

            if not options and type(cur).__name__ != "list":
                # the options list is blank. we only care about matching the input type to the current type
                inpt_request = "Input new parameter value of type " +type(cur).__name__ + " or type info for hyperparameter description: "
            elif not options:
                # it's a list
                list_type = type(cur[0]).__name__
                inpt_request = "Inpt a value of type " + list_type +" to add to the list or type info for hyperparameter description or press enter to finish entering list values: "
                
                hyperparam_df.at[cur_index, 'current_value'] = []
                print("\nOriginal value", param, "was", cur)
                cur = []

            else:
                type_list = [x.__name__ for x in get_types_from_list(options)]
                if type_list:
                    # if there's more than just strings in the options list
                    options = [x for x in options if type(x).__name__ == "str"]
                    inpt_request = "Input new parameter from the options: " +", ".join(options) + ". Or a value of type " + ", ".join(type_list)+ ". Or type info for hyperparameter description: "
                else:
                    #only strings. don't need to list types
                    inpt_request = "Input new parameter from the options: " +", ".join(options) + ". Or type info for hyperparameter description: "

            while True:
                print("\nCurrent value for", param, "is", cur)
                inpt = read_input(input(inpt_request))
                if (not inpt) and list_type:
                    break
                elif inpt == "info":
                    print("\n",hyperparam_df.at[cur_index, 'info'])
                elif (not options and type(inpt) == type(cur)) or (options and inpt in options) or (type_list and type(inpt).__name__ != "str" and type(inpt).__name__ in type_list):
                    # options is blank and the type of the input matches the current value or
                    # the input value matches something in the options list (for string options) or
                    # the input isn't a string but it's type in on the type list
                    hyperparam_df.at[cur_index, 'current_value'] = inpt
                    break
                elif list_type and type(inpt).__name__ == list_type:
                    cur.append(inpt)
                    hyperparam_df.at[cur_index, 'current_value'] = cur
                else:
                    print("\n",inpt,"is not an available option. Please try again.")
    print()

    return hyperparam_df