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


def prepare_hdf5():
    hdf5_file_path = args.hdf5_file
    output_folder = args.result_image_folder

    # 저장 폴더가 없으면 생성
    os.makedirs(output_folder, exist_ok=True)

    # HDF5 파일 열기
    with h5py.File(hdf5_file_path, 'r') as file:
        dataset_key = 'images'
        if dataset_key not in file:
            raise KeyError(f"HDF5 파일에서 '{dataset_key}' 데이터셋을 찾을 수 없습니다.")

        # 데이터셋에서 이미지 데이터 가져오기
        images_data = file[dataset_key][:]

    images_data = images_data.transpose(0, 2, 3, 1)

    return output_folder, images_data


def save_png(output_folder, images_data, index_list):
    # 이미지 추출 및 저장
    for i, image_idx in enumerate(tqdm(index_list)):
        # 이미지 데이터를 numpy 배열로 변환
        image_data = images_data[image_idx]
        image_array = np.array(image_data, dtype=np.uint8)

        # numpy 배열을 이미지로 변환
        image_pil = Image.fromarray(image_array)

        # 이미지를 PNG 형식으로 저장
        output_path = os.path.join(output_folder, f'image_{i + 1}.png')  # 인덱스 번호를 이미지 번호로 변경
        image_pil.save(output_path)

        # print(f"이미지 {image_idx + 1}을 {output_path}에 저장했습니다.")


def make_image_caption_pair(df, index_list):
    image_caption_list = []

    for i in range(len(index_list)):
        image_caption_dict = {'file_name': f'image_{i+1}.png', 'text': df.iloc[i, 1]}
        image_caption_list.append(image_caption_dict)

    return image_caption_list

def save_image_caption_pair(image_caption_list):
    data_root_path = args.result_image_folder

    with open(data_root_path + "metadata.jsonl", 'w') as f:
        for item in image_caption_list:
            f.write(json.dumps(item) + "\n")

    return data_root_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare text data')
    parser.add_argument('--hdf5_file', type=str, default="data/test_data/TEST_IMAGES.hdf5", help='Path to HDF5 file')
    parser.add_argument('--caption_csv_file', type=str, default="data/test_data/caption/test_caption_data_drop_duple.csv", help='Path to Excel file')
    parser.add_argument('--result_image_folder', type=str, default="data/test_data/image/", help='Path to store image folder')
    args = parser.parse_args()
    
    df, index_list = prepare_text(args.caption_csv_file)
    # print(len(index_list)) # 중복 제거 후 남은 인덱스 리스트

    output_folder, images_data = prepare_hdf5() # hdf5 데이터셋에서 이미지 데이터 가져오기

    save_png(output_folder, images_data, index_list) # 이미지 데이터를 png 형식으로 저장

    image_caption_pair = make_image_caption_pair(df, index_list)
    data_root_path = save_image_caption_pair(image_caption_pair)
    print(data_root_path)

