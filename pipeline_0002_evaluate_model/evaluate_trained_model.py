import torch
import os, sys
import argparse

from PIL import Image
from transformers import BlipForConditionalGeneration
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

abs_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.dirname(abs_path))

from pipeline_0000_prepare_data.prepare_data import prepare_text
from pipeline_0001_train_model.train_model_blip import load_processor_and_model


# Load the processor and the trained model
def load_processor_and_trained_model(result_checkpoint_file):
    processor, _, device, _ = load_processor_and_model()
    model = BlipForConditionalGeneration.from_pretrained(result_checkpoint_file)
    model.to(device)

    return processor, model, device


# Load the existing caption dataset(reference_caption) and convert it into a list
def convert_list_reference_caption(df):
    reference_caption = df.Sentence.to_list()

    return reference_caption


# Input images into the trained model to generate a caption
def generate_caption_trained_model(data_root_path, reference_caption, processor, model, device):
    generated_caption = []

    for i in range(len(reference_caption)):
        output_path = os.path.join(data_root_path, f'image_{i + 1}.png')
        image = Image.open(output_path).convert("RGB")

        inputs = processor(image, return_tensors="pt").to(device, torch.float16)

        generated_ids = model.generate(**inputs, max_new_tokens=20)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        generated_caption.append(generated_text)

        if not i%100:
            print(i)

    return generated_caption


# Compare the reference caption with the generated caption to calculate the BLEU score
def evaluate_bleu(reference_caption, generated_caption):
    reference_token_lists = [[ref.split()] for ref in reference_caption]
    candidate_token_lists = [cand.split() for cand in generated_caption]

    # Brevity Penalty: A technique that corrects the issue arising when the candidate(generated) sentence is shorter than the reference sentence
    smooth_func = SmoothingFunction().method1
    bleu_score = corpus_bleu(reference_token_lists, candidate_token_lists, smoothing_function=smooth_func, weights=(0.25, 0.25, 0.25, 0.25))
    
    return bleu_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare image data')
    parser.add_argument('--image_folder', type=str, default="data/test_data/image/", help='Path to Image Folder')
    parser.add_argument('--result_checkpoint_file', type=str, default='data/checkpoints/test_blip_fine_tuning_model.pt', help='Path to store trained model')
    parser.add_argument('--caption_csv_file', type=str, default="data/test_data/caption/test_caption_data_drop_duple.csv", help='Path to Excel file')
    args = parser.parse_args()

    data_root_path = args.image_folder
    result_checkpoint_file = args.result_checkpoint_file
    caption_csv_file = args.caption_csv_file

    processor, model, device = load_processor_and_trained_model(result_checkpoint_file)
    df = prepare_text(caption_csv_file)
    reference_caption = convert_list_reference_caption(df)
    generated_caption = generate_caption_trained_model(data_root_path, reference_caption, processor, model, device)

    bleu_score = evaluate_bleu(reference_caption, generated_caption)
    print(f'BLEU score : {bleu_score}')
