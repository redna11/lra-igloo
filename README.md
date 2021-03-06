# [LRA-IGLOO]("https://github.com/redna11/lra-igloo")
[![Python](https://img.shields.io/pypi/pyversions/tensorflow.svg?style=plastic)](https://badge.fury.io/py/tensorflow)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

## Expected results

|               | ListOps   | Text-IMDB | Retrieval | Image-cifar10 | Path      | Avg    |
--------------- | --------- | --------- | --------- | ------------- | --------- | ------ |
Text Accuracy   | 39        | 84.5      | 75.5      |        47     | 76.5      | 64.5   |
Params baseline | 19.9M     | 3.5M      | 1.087k    |     	380k    | 446k      |        |
Params IGLOO    | 18.3M     | 3.50M     | 1.145k    |     	132k    | 440k      |        |

Those are the results one is expected to find using IGLOO on the [Long Range Arena](https://github.com/google-research/long-range-arena) suite of benchmarks.

```
@inproceedings{
tay2020long,
title={Long Range Arena : A Benchmark for Efficient Transformers },
author={Yi Tay, Mostafa Dehghani, Samira Abnar, Yikang Shen, Dara Bahri,
Philip Pham, Jinfeng Rao, Liu Yang, Sebastian Ruder, Donald Metzler},
booktitle={ArXiv Preprint},
year={2020},
url={},
note={under review}
}
```

## Prerequisites
    Tensorflow 2.0.0
    Tensorflow-datasets 2.0.0
    Numpy

## Building
To install the prerequisites.
```
$ pip install -r requirements.txt
```

## Running

```
$ python longrangearena_cifar10.py
```

```
$ python longrangearena_imdb.py
```

```
$ python longrangearena_matching.py
```

```
$ python pathfinder/longrangearena_pathfinder.py
```

```
$ python lra_listops/longrangearena_listops.py
```
