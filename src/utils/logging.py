import json

class LoggingStruct:
    def __init__(self):
        """A simple logging structure to keep track of training metrics over time and save them to a JSON file."""
        self.log = {}

    def update(self, data: dict):
        """Update the log with new data.

        Args:
            data (dict): A dictionary of metric names and their corresponding values to log.
        """
        self.log.update(data)