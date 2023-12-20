### Prepare data

### Setting anaconda env
```
CONTAINER_NAME="pipeline_0000-prepare_data"
conda create -n ${CONTAINER_NAME} -c conda-forge python=3.10 cudatoolkit=11.8
conda activate ${CONTAINER_NAME}
conda install -c conda-forge cudnn=8.8.0

pip install -r requirements.txt
```

### Usage
```
python prepare_data.py \
    --hdf5_file ../data/test_data/TEST_IMAGES.hdf5 \
    --caption_csv_file ../data/test_data/caption/test_caption_data_drop_duple.csv \
    --result_image_folder ../data/test_data/image/

```