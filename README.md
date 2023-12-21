# MLOps Project

# Table of Contents

1. [Objective](#objective)
	- [What I Learned from the Project](#what-i-learned-from-the-project)
	- [Thanks to](#thanks-to)
2. [Pipeline Configuration](#pipeline-configuration)
3. [Usage](#usage)
	- [Prerequisites](#prerequisites)
	- [Data](#Data)

## Objective

- This project is aimed at creating an MLOps pipeline for training an image captioning model for diffusion model learning.

### What I Learned from the Project

- Comprehensive processes and related programs & techniques in MLOps
	- Docker
	- Kubeflow & Kubernetes
	- AWS
	- Setting up a tunneling server using Ngrok, AWS EC2 as a reverse proxy
	- Various image captioning models (examples)
		- BLIP1
		- BLIP2
		- GIT
		- etc...
	- Gradio (used for deploying the model)

### Thanks to...

- [Codime](https://codime.io/)

---

## Pipeline Configuration

The current pipelines in use:

1. [**pipeline_0000_prepare_data**](#https://github.com/JeongYuH/MLOps_Pipeline/tree/main/pipeline_0000_prepare_data)
    - A pipeline for preprocessing data, specifically tailored for the diffusion model learning image captioning.
    - Tasks include data loading, handling missing values, feature engineering, and saving processed data.
	- This section uses ECCV data preprocessing, and detailed explanations for this process can be found in the README.md file in the respective folder.

2. [**pipeline_0001_train_model**](#https://github.com/JeongYuH/MLOps_Pipeline/tree/main/pipeline_0001_train_model)
    - A pipeline designed for training the image captioning model within the context of diffusion model learning.
    - Involves loading preprocessed data, training the image captioning model, and saving the trained model.

3. [**pipeline_0002_evaluate_model**](#https://github.com/JeongYuH/MLOps_Pipeline/tree/main/pipeline_0002_evaluate_model)
    - A pipeline dedicated to evaluating the trained image captioning model for diffusion model learning.
    - Tasks include loading test data, evaluating the image captioning model, and saving the results.

4. [**pipeline_0003_deploy_model**](#https://github.com/JeongYuH/MLOps_Pipeline/tree/main/pipeline_0003_deploy_model)
    - A pipeline focused on deploying the trained image captioning model tailored for diffusion model learning.
    - Involves loading the trained model, deploying it, and configuring the environment.

---

# Usage

## Prerequisites

- To proceed with this project, you will need the following programs:
	- Kubeflow
	- Docker
	- miniconda3
	- Git & GitHub

## Data

- [ECCV Data](https://github.com/xuewyang/Fashion_Captioning)

- [HuggingFace_jinaai/de](https://huggingface.co/datasets/jinaai/fashion-captions-de)

- [Kaggle](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset/data)

- Codime Own Dataset



