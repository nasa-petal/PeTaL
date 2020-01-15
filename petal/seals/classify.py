import sys, pickle, pandas

NUM_REQ_ARGS = 3
REQ_MESSAGE = "Inappropriate number of arguments. Required arguments are:\n path to test data\n path to model (from train.py)\n desired path for csv output"

def main():
    if len(sys.argv) - 1 < NUM_REQ_ARGS:
        print(REQ_MESSAGE)
        return

    data_path = str(sys.argv[1])
    pickle_path = str(sys.argv[2])
    output_path = str(sys.argv[3])

    data = pandas.read_csv(data_path)
    model = pickle.load( open(pickle_path, "rb" ) )
    data = model.encode_categorical(data)

    labels = model.classify_data(data).tolist()

    if model.supervised:
        data['class'] = labels   
        data = model.decode_categorical(data)
    else:
        data = model.decode_categorical(data)
        data['class'] = labels

    data.to_csv(output_path,index=False)
    
if __name__ == "__main__":
    main()   