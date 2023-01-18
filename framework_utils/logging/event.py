"""
Event class implementation for the purpose of subscribable logging.
"""


class Event(list):
    """
    Event class, to which functions can be appended. If the event is "fired" (called),
    every function will be called in the order that they were appended. This class imitates
    the publisher-subscriber pattern.
    """
    def __call__(self, *args, **kwargs):
        _ = [event_handler(*args, **kwargs) for event_handler in self]

    def __repr__(self):
        return "Event(%s)" % list.__repr__(self)


class TrainStatusChangedEventGroup:
    """
    Event group, these functions are fired during the training and validation process of a model.
    """
    def __init__(self):
        # Run level handlers
        self.on_train_start = Event()
        self.on_train_end = Event()
        # Epoch level handlers
        self.on_epoch_start = Event()
        self.on_epoch_end = Event()
        self.on_epoch_train_start = Event()
        self.on_epoch_train_end = Event()
        self.on_epoch_val_start = Event()
        self.on_epoch_val_end = Event()
        # Batch level handlers
        self.on_batch_start = Event()
        self.on_batch_end = Event()

