from petal.pipeline.module_utils.module import Module

from pprint import pprint

class MockMLModel(Module):
    def __init__(self, in_label='MLData', out_label=None, connect_labels=None, name='MockMLData', epochs=2,
                       train_fraction=0.8, test_fraction=0.15, validate_fraction=0.05):
        Module.__init__(self, in_label=in_label, out_label=out_label, connect_labels=connect_labels, name=name, page_batches=True)

        self.epochs = epochs

        self.validate_fraction = validate_fraction
        self.train_fraction    = train_fraction
        self.test_fraction     = test_fraction

    def process(self, node, driver=None):
        pass

    def train(self, batch):
        print('Training on ', batch.uuid, flush=True)

    def test(self, batch):
        print('Testing on ', batch.uuid, flush=True)

    def val(self, batch):
        print('Validating on ', batch.uuid, flush=True)

    def process_batch(self, batch, driver=None):
        if batch.rand < self.train_fraction:
            self.train(batch)
        elif batch.rand < self.train_fraction + self.test_fraction:
            self.test(batch)
        else:
            self.val(batch)

