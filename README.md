# Maldi-Tools

This repository contains scripts for extracting images from the TIMS TOF MALDI 2 with a user defined target library.

## Installation Instructions


### Conda
The fastest way to get up and running with **Maldi Tools** is to create a `conda` environment. You will need any flavor of `conda`. Some options include:
- [Miniforge](https://github.com/conda-forge/miniforge) (**Recommended** as it contains `conda-forge`)
- [Anaconda](https://www.anaconda.com/products/distribution) (also works)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (also works)

### Install Maldi-Tools

Once you have any one of these installed, clone the repository with:

```sh
git clone [maldi-tools](https://github.com/angelolab/maldi-tools.git)
```

Then change the directory to `maldi-tools` and create the environment:

```sh
conda env create -f environment.yml
```

This will install all the dependencies and the most up-to-date version of `maldi-tools`.

## Using Maldi-Tools

Maldi-Tools can be started by first activating the Conda environment with:

```sh
conda activate maldi-pipeline
```

and then the Jupyter interface can be started with:

```sh
jupyter lab
```

## Development

Maldi-Tools is a *poetry* project and development requires [`python-poetry`](https://python-poetry.org) and [`pre-commit`](https://pre-commit.com).

We recommend installing both with [`pipx`](https://pypa.github.io/pipx/).

For development, after cloning the repository, create the development environment and install dependencies for `maldi-tools`, development dependencies and testing dependencies with:
```sh
poetry install --with dev,test
```

Next, install the repo's `pre-commit` hooks with:
```sh
pre-commit install --install-hooks
```
Pre-commit hooks will sort imports, fix formatting, and lint your code for you. They run on each commit, but you may also run them manually with:

```sh
pre-commit run --all-files
```

Tests can be run with:
```sh
poetry run pytest
```

### Issues?
Maldi-Tools is in early, active development. Please submit an issue for any bugs, or ideas for feature requests.
