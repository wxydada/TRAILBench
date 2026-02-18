# TRAILBench

## Introduction

This is the implementation for TRAILBench: Personalized Function Calling Benchmark Grounded in Real-world User Histories, the first comprehensive benchmark for personalized function calling built on real interaction data. It includes a high-quality dataset of anonymized interaction histories and an evaluation suite with controlled query vagueness.

Fold appendix include the Dataset Details and Implemention details in our paper.

## Quick Start



### Installation

```bash
conda create -n TRAILBench python=3.10
conda activate TRAILBench

git clone https://github.com/wxydada/TRAILBench.git

cd TRAILBench

pip install -r requirements.txt
```

## Evaluating Open-Source Model


### Set Up


set up your basic config in [`config.yaml`](config.yaml) . For more specific parameters, you can change the parameters in [`config.yaml`](predict_open.py).

### run evaluation


```bash
python predict_open.py
```
You can find the predict results in `output_dir` repository.

## Evaluating Close-source(API) models


### Set Up

set up your api key and base url in [`config.yaml`](config.yaml). For more specific parameters, you can change the parameters in [`config.yaml`](predict_open.py).

### run evaluation


```bash
python predict_api.py
```
You can find the predict results in `output_dir` repository.


