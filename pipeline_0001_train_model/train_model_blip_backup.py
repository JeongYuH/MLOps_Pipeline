import torch
import argparse
import os, sys

from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, BlipForConditionalGeneration

# 상대 경로(현재 위치한 디렉토리 기준) -> 절대 경로(시스템 전체 디렉토리 구조를 기준)
# 스크립트 파일이 위치한 디렉토리의 부모 디렉토리를 모듈 검색 경로에 추가
abs_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.dirname(abs_path))

from prepare_data_01 import prepare_data


def load_processor_and_model():
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    device = torch.device("cuda:0")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    return processor, model, device, optimizer


class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoding = self.processor(images=item["image"], text=item["text"], padding="max_length", return_tensors="pt")
        encoding = {k: v.squeeze() for k, v in encoding.items()}

        return encoding


def train_model(model, train_dataloader, optimizer, epochs):
    epoch_losses = []

    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0

        for batch in train_dataloader:
            input_ids = batch.pop("input_ids").to(device)
            pixel_values = batch.pop("pixel_values").to(device)

            outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=input_ids)
            loss = outputs.loss
            epoch_loss += loss.item()

            print('loss :', loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss /= len(train_dataloader)
        epoch_losses.append(epoch_loss)
        print(f"Epoch {epoch+1}: Average Loss: {epoch_loss}")

    return epoch_losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare image data')
    parser.add_argument('--image_folder', type=str, default="data/test_data/image/", help='Path to Image Folder')
    parser.add_argument('--excel_file', type=str, default='data/test_data/caption/test_caption_data_drop_duple.csv', help='Path to Excel file')
    args = parser.parse_args()

    data_root_path = args.image_folder

    processor, model, device, optimizer = load_processor_and_model()
    dataset = prepare_data.make_dataset_from_metadata(data_root_path)
    image_captioning_dataset = ImageCaptioningDataset(dataset, processor)
    train_dataloader = DataLoader(image_captioning_dataset, shuffle=True, batch_size=2)
    epoch_losses = train_model(model, train_dataloader, optimizer, epochs=5)

    model.save_pretrained('test_blip_fine_tuning_model.pt')


# conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch -c nvidia

# python train_model_02/train_model_blip.py



# def prepare_image_text(excel_file_path):
#     captions = []
#     df = pd.read_csv(excel_file_path)

#     for i in range(len(df)):
#         caption_dict = {'file_name': f'image_{i+1}.png', 'text': df.iloc[i, 1]}
#         captions.append(caption_dict)

#     root = args.image_folder

#     with open(root + "metadata.jsonl", 'w') as f:
#         for item in captions:
#             f.write(json.dumps(item) + "\n")

#     dataset = load_dataset("imagefolder", data_dir=root, split="train")

#     return dataset


# def prepare_image_text(excel_file_path):
#     image_caption_pair = make_image_caption_pair(excel_file_path)
#     data_root_path = save_image_caption_pair(image_caption_pair, args.image_folder)
#     dataset = make_dataset_from_metadata(data_root_path)

#     return dataset