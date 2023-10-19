# Minimal Working Example (MWE) Execuotr

This is a [SegServe](https://github.com/hip-satomi/SegServe) executor provides a minimal working example (MWE) and demonstrates how to add custom executors into the ObiWan-Microbi software.

## Getting started

The general idea of the executors is that we encapsulate the segmentation process into `mlflow projects`. Thus, all dependencies are collected in a [`conda.yaml`](conda.yaml) and access interfaces are defined in [MLProject](MLProject).

`mlflow` is then used to download a specific version of this repository from its github url and executes the segmentation method on a list of images (see [test cases](tests/test.py)).

### 1. Defining software dependencies

In order to list all software dependencies please edit the [conda.yaml](conda.yaml) file. It is best practice to fix the software package versions to minimize problems occuring due to new releases.

For deep-learning dependencies, like `pytorch` and `tensorflow` please have a look at their installation insturctions and add the dependencies or additional channels to the [conda.yaml](conda.yaml) file. You can also find examples in our exsting executor [repositories](https://github.com/hip-satomi).

### 2. Integrate custom segmentation method

The main segmentation work is performed in the [main.py](main.py) and there in step (2). For every image a segmentation method can be called, e.g. utilizing deep-learning tools like pytorch or tensorflow in order to obtain individual contours of objects.

These instance segmentation objects are then saved into a json format and provided to segServe. Therefore, these executors are fully independent, manage their own dependencies and only need to adhere to the segmentation storing format.

### Advanced usage

Advanced usages include, e.g., multiple segmentation method implementations in a single executor repository. This can be achieved by completely different implementations or by custom parameters that can be passed. For this you need to edit the [MLProject](MLProjcet) file and add new entrypoints (see [mlflow documentation](https://mlflow.org/docs/latest/projects.html)).

## Testing

## Local testing

Make sure you have [anaconda](https://www.anaconda.com/products/distribution) installed and an active environment with [`mlflow`](https://pypi.org/project/mlflow/). Then execute
```bash
pip install mlflow
mlflow run ./ -e main -P input_images=<path to your local image or image folder (*.png)>
```
and replace `<path to your local image or image folder (*.png)>` by a path to an image or a folder of images. For a folder, all images need to be present in png format but this behavior can be changed in [main.py](main.py). For specifying custom parameters please have a look into the [mlflow documentation](https://mlflow.org/docs/latest/projects.html).

The resulting segmentation should be written to `output.json` and logged as an artifact in the mlflow run.

## Unit testing

