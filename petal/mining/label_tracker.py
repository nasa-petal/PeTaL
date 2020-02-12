class LabelTracker():
    '''
    Used for scheduling dependent modules withing scheduler
    '''
    def __init__(self):
        self.tracker = dict()
        self.throttle_count_dict = dict()

    def count(self, label):
        if label not in self.tracker:
            return 0
        else:
            return len(self.tracker[label])

    def throttle_count(self, label):
        if label not in self.throttle_count_dict:
            return 100
        else:
            return self.throttle_count_dict[label]

    def set_throttle_count(self, label, n):
        self.throttle_count_dict[label] = n

    def add(self, label, uuid):
        for sublabel in label.split(':'):
            if sublabel in self.tracker:
                self.tracker[sublabel].add(uuid)
            else:
                self.tracker[sublabel] = {uuid}

    def get(self):
        return self.tracker

    def clear(self, label):
        self.tracker[label].clear()
