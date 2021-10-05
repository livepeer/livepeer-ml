# About
This repository contains machine learning models used by LivePeer services, corresponding code and DevOps scripts.

# Installation
Pre-requisites:
- Python 3.9+
- PIP 19.0+ 
```
pip install -r requirements.txt
```

# Structure
Directories contain the code and corresponding models.

# Dataset access
Easiest way to access the dataset is to mount GCS bucket with [gcsfuse](https://github.com/GoogleCloudPlatform/gcsfuse/). However, for training and testing it's recommended to synchronize the dataset to local file system (look out for -d option):
```
gsutil -m rsync -d -r gs://livepeer-ml-data data
```

# Models

## Content classification
This model is based on [MobileNet v2](https://arxiv.org/abs/1801.04381) architecture. Its purpose is to identify unwanted content during transcoding.