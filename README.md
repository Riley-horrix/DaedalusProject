# DaedalusProject

Testing environment for reinforcement based agents for controlling aircraft.

## Running

To run from the base directory, first ensure that my fork of [NeuralPlane](https://github.com/Riley-horrix/NeuralPlane) repository is installed in `./lib/` and that it is checked out on the `thesis` branch.

Then install dependencies with `pip install -r requirements.txt`.

To run, first edit `src/scripts/runner.py` to load the correct configuration. Then execute `python -m src.scripts.runner --data-path=./~path-to-data~`, to run.