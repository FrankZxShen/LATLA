import datetime
import argparse
import logging
import logging.handlers
import os
import sys
import json
from tqdm import *

import requests
import torch
from PIL import Image
from pythia.utils.llava_gen import disable_torch_init, llava_infer
from llava.model.builder import load_pretrained_model
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from llava.model import *
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
# fork from https://github.com/haotian-liu/LLaVA

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-vlm', '--llava', default='/DATACENTER1/szx/tap/data/models/llava-v1.5-7b/', help='name of t5lm model to use')
    parser.add_argument('-c', '--caption', default='/DATACENTER1/szx/tap/data/mid_json/texvqa_caption_gen.json', help='path to caption file')
    parser.add_argument('-cv', '--captionvlm', default='/DATACENTER1/szx/tap/data/mid_json/texvqa_caption_vlm_gen.json', help='path to vlm caption file')
    parser.add_argument("--default-path", type=str, default='/DATACENTER1/szx/tap/')
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--load-8bit", type=bool, default=False)
    parser.add_argument("--load-4bit", type=bool, default=True)
    args = parser.parse_args()

    disable_torch_init()

    model_name = args.llava.split('/')[-2]
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.llava, args.model_base, model_name, args.load_8bit, args.load_4bit)

    all_parameters = []
    with open(args.captionvlm,'w') as captionvlm_gen_file:
            captionvlm_gen_file.write('\n')
    with open(args.caption,'r') as caption_file:
        for img_captions in caption_file:
            if img_captions == {}:
                continue
            img_caption = json.loads(img_captions)
            all_parameters.append(img_caption)
    disable_tqdm = False
    for single_parameters in tqdm(all_parameters, disable=disable_tqdm):
        if single_parameters == {}:
            continue
        ocr_cap_in = single_parameters["ocr_tokens"]
        
        # ocr_word_list = ocr_cap_in.split(',')
        # ocr_cap_in = ', '.join(['"{}"'.format(word) for word in ocr_word_list])
        single_image = load_image(args.default_path + single_parameters["image_path"])

        # caption = {}
        cap = {}
        # single caption
        cap["image_path"] = single_parameters["image_path"]
        cap["blip_captions"] = single_parameters["captions"]
        cap["vlm_captions"] = llava_infer(single_image, ocr_cap_in, tokenizer, model, image_processor, model_name)
        while len(cap["vlm_captions"]) >= 1024:
            ocr_cap_in = ''
            cap["vlm_captions"] = llava_infer(single_image, ocr_cap_in, tokenizer, model, image_processor, model_name)
        cap["ocr_tokens"] = ocr_cap_in
        cap["question"] = single_parameters["question"]
        cap["candidates"] = single_parameters["candidates"]
        cap["pred_answer"] = single_parameters["pred_answer"]

        print("*********************************************")
        print(cap)
        with open(args.captionvlm,'a+') as vlm_captions_gen_file:
            json.dump(cap,vlm_captions_gen_file)
            vlm_captions_gen_file.write('\n')



if __name__ == "__main__":
    main()
