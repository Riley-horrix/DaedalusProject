import json

class Config:
    def __init__(self, name: str):
        self.name = name
        self.data = {}

    def __call__(self, key: str, default=None):
        """Allows the config to be accessed like a dictionary, e.g., config['learning_rate'].

        Args:
            key (str): The key of the configuration parameter to access.
            default: The default value to return if the key is not found.
            Returns:
            The value of the configuration parameter.
        """
        if key not in self.data:
            if default is not None:
                print(f"Warning: Configuration key '{key}' not found for {self.name}. Returning default value: {default}")
                return default
            raise KeyError(f"Configuration key '{key}' not found for {self.name}.")

        return self.data[key]

    def load_from_file(self, path: str):
        """Loads the configuration from a json file.

        Args:
            path (str): Path to the configuration file.
        """
        with open(path, 'r') as f:
            self.data = json.load(f)
            if 'name' not in self.data or self.data['name'] != self.name:
                raise ValueError(f"Configuration file name '{self.data.get('name', None)}' does not match expected name '{self.name}' for config {path}.")

    def save_to_file(self, path: str):
        """Saves the configuration to a json file.

        Args:
            path (str): Path to the configuration file.
        """
        with open(path, 'w') as f:
            self.data['name'] = self.name
            json.dump(self.data, f)