# Evaluate model

# Setting conda env

```
CONTAINER_NAME="pipeline_0002_evaluate_model"
conda create -n ${CONTAINER_NAME} -c conda-forge python=3.10 cudatoolkit=11.8
conda activate ${CONTAINER_NAME}
conda install -c conda-forge cudnn=8.8.0

pip install -r requirements.txt

pip install numpy==1.24.1 notebook==6.5.5 traitlets==5.9.0
```

# Usage

```
python evaluate_trained_model.py \
    --image_folder ../data/test_data/image \
    --caption_csv_file ../data/test_data/caption/test_caption_data_drop_duple.csv \
    --result_checkpoint_file ../data/checkpoints/test_blip_fine_tuning_model.pt
```
