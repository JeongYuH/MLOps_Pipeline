import argparse
import pandas as pd
import re
import h5py
import json
import numpy as np
import os

from PIL import Image

from tqdm import tqdm


def remove_special_tokens(text):
    cleaned_text = re.sub(r'<start>|<end>|<pad>', '', text).strip()
    return cleaned_text

def prepare_text(caption_csv_file):
    df = pd.read_csv(caption_csv_file)
    df = df.rename(columns={'Unnamed: 0':'num'})
    for i in range(len(df)):
        df.iloc[i, 1] = remove_special_tokens(df.iloc[i, 1])

    index_list = df.num.tolist()

    return df, index_list


def prepare_hdf5(hdf5_file_path, result_image_folder):
    os.makedirs(result_image_folder, exist_ok=True)

    # Open HDF5 File
    with h5py.File(hdf5_file_path, 'r') as file:
        dataset_key = 'images'
        if dataset_key not in file:
            raise KeyError(f"HDF5 파일에서 '{dataset_key}' 데이터셋을 찾을 수 없습니다.")

        # Importing image data from a dataset
        images_data = file[dataset_key][:]

    images_data = images_data.transpose(0, 2, 3, 1)

    return result_image_folder, images_data


def save_png(output_folder, images_data, index_list):
    for i, image_idx in enumerate(tqdm(index_list)):
        # Converting image data to a numpy array
        image_data = images_data[image_idx]
        image_array = np.array(image_data, dtype=np.uint8)

        # Converting a numpy array to an image
        image_pil = Image.fromarray(image_array)

        # Save the image in PNG format
        output_path = os.path.join(output_folder, f'image_{i + 1}.png')
        image_pil.save(output_path)


def make_image_caption_pair(df):
    image_caption_list = []

    for i in range(len(df)):
        image_caption_dict = {'file_name': f'image_{i+1}.png', 'text': df.iloc[i, 1]}
        image_caption_list.append(image_caption_dict)

    return image_caption_list

def save_image_caption_pair(image_caption_list, data_root_path):
    with open(data_root_path + "metadata.jsonl", 'w') as f:
        for item in image_caption_list:
            f.write(json.dumps(item) + "\n")

    return data_root_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare text data')
    parser.add_argument('--hdf5_file_path', type=str, default="data/test_data/TEST_IMAGES.hdf5", help='Path to HDF5 file')
    parser.add_argument('--caption_csv_file', type=str, default="data/test_data/caption/test_caption_data_drop_duple.csv", help='Path to Excel file')
    parser.add_argument('--result_image_folder', type=str, default="data/test_data/image/", help='Path to store image folder')
    args = parser.parse_args()

    caption_csv_file = args.caption_csv_file
    hdf5_file_path = args.hdf5_file_path
    result_image_folder = args.result_image_folder
    
    # Preparing images and text
    df, index_list = prepare_text(caption_csv_file)
    output_folder, images_data = prepare_hdf5(hdf5_file_path, result_image_folder)
    save_png(output_folder, images_data, index_list)

    # Image and Caption Pairing
    image_caption_pair = make_image_caption_pair(df)
    data_root_path = save_image_caption_pair(image_caption_pair, result_image_folder)
