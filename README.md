# ESAE

## Installation

Please ensure that you have [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) installed on your system, as it is required for managing dependencies in this project.

To create and activate the Conda environment using the provided `environment.yml` file, please run the following command:

```sh
conda env create -f environment.yml
conda activate esae
```

Before proceeding, please update the system paths specified in `source/__init__.py` to match your configuration. These paths are used for storing the datasets, model checkpoints, and many others.

```python
from pathlib import Path

workspace = Path("/data/user_data/haok/esae")
workspace.mkdir(mode=0o770, parents=True, exist_ok=True)

import os

os.environ["HF_HOME"] = "/data/user_data/haok/huggingface"
```

## Overview

To train a Sparse Autoencoder (SAE), the first step is to download the dataset and compute the embeddings that will later be reconstructed. Initialize all datasets in this repository using the following command:

```sh
python3 -m source.dataset.msMarco
```

### Running Experiments

To experiment with different hyperparameters or model configurations, refer to the experiments in the `source/model/archive` directory. Each file in this directory contains a specific model setup.

If you want to create a new experiment, you can do so by adding a new file with your desired hyperparameters and configurations. Once ready, run the following command, replacing {version} with your file name:

```sh
python3 -m source.model.{version}
```

For example, if your new experiment file is under `source/model/240825A.py`, you would run:

```sh
python3 -m source.model.240825A
```

Model checkpoints are automatically saved under `{workspace}/model/{version}/state/`, where workspace is the path specified in `source/__init__.py`. This makes it easy to manage and retrieve your experiment results.

### Evaluating Performance

## Quality Assurance

### Standardized Interface

To ensure a clean and reusable codebase, this repository follows best practices by defining the interfaces in `source/interface.py`. All core components implement standardized interfaces that promote consistency and modularity. For instance, the Dataset class defines a blueprint that any dataset must follow by implementing the didIter method. This method enables the client to iterate over all document IDs in batches.

Here's an example:

```python
from abc import ABC, abstractmethod
from typing import Iterator, List

class Dataset(ABC):
    name: DatasetName

    @abstractmethod
    def didIter(self, batchSize: int) -> Iterator[List[int]]:
        """
        Iterate over the document IDs.

        :param batchSize: The batch size for each iteration.
        :return: The iterator over the document IDs.
        """
        raise NotImplementedError
```

### Testing Locally

To ensure matainability, this codebase is fully type-checked using mypy and thoroughly tested with pytest. As new components are integrated into the interface, please ensure that corresponding test cases are added. Place your test cases under the relevant directories to keep the test suite comprehensive and organized.

You can run the following commands to perform these checks:

```sh
mypy source
pytest source
```
