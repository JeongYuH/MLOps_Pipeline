import torch
import argparse
import os, sys

from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, BlipForConditionalGeneration
from datasets import load_dataset


def make_dataset_from_metadata(data_root_path):
    dataset = load_dataset("imagefolder", data_dir=data_root_path, split="train")
    return dataset


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

    return model, epoch_losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare image data')
    parser.add_argument('--image_folder', type=str, default="data/test_data/image/", help='Path to Image Folder')
    parser.add_argument('--result_checkpoint_file', type=str, default='data/checkpoints/test_blip_fine_tuning_model.pt', help='Path to store trained model')
    args = parser.parse_args()

    data_root_path = args.image_folder
    result_checkpoint_file = args.result_checkpoint_file

    processor, model, device, optimizer = load_processor_and_model()
    dataset = make_dataset_from_metadata(data_root_path)
    image_captioning_dataset = ImageCaptioningDataset(dataset, processor)
    train_dataloader = DataLoader(image_captioning_dataset, shuffle=True, batch_size=2)
    model, epoch_losses = train_model(model, train_dataloader, optimizer, epochs=5)

    model_dirname = os.path.dirname(result_checkpoint_file)
    os.makedirs(model_dirname, exist_ok=True)
    model.save_pretrained(result_checkpoint_file)


# conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch -c nvidia
