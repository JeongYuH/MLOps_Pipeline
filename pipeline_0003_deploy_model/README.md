# Deploy model

# Setting conda env

```
CONTAINER_NAME="pipeline_0003_deploy_model"
conda create -n ${CONTAINER_NAME} -c conda-forge python=3.10 cudatoolkit=11.8
conda activate ${CONTAINER_NAME}
conda install -c conda-forge cudnn=8.8.0

pip install -r requirements.txt
```

# Usage

```
python app.py
```

---
title: Comparing Captioning Models
emoji: 🔥
colorFrom: yellow
colorTo: pink
sdk: gradio
sdk_version: 4.5.0
app_file: app.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference


