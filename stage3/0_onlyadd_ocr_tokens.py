import numpy as np
import pandas as pd
import json

def get_prompts(val_line,val_stvqa,is_stvqa):
    Path = []
    num_ocr = []
    ocr_coordinates = []
    Question = []
    Ocr = []
    Image_class = []
    gt = []
    if not is_stvqa:
        for train in val_line:
            if len(train.keys()) < 10:
                continue
            Question.append(train['question'])
            Path.append(train['image_path'])
            ocr_cap_in = ", ".join([x.lower() for x in train['ocr_tokens']])
            Ocr.append(ocr_cap_in)
    else:
        for train in val_stvqa:
            if len(train.keys()) < 10:
                continue
            Question.append(train['question'])
            Path.append(train['image_path'])
            ocr_cap_in = ", ".join([x.lower() for x in train['ocr_tokens']])
            Ocr.append(ocr_cap_in)

    
    return Ocr, Question, Path

is_stvqa=False

val_json_path = '/home/szx/project2/TAP/data/mid_json/VQAModel_out_TextVQA_val_1217_2.json'
# val_json_path = '/home/szx/project2/TAP/data/mid_json/VQAModel_out_STVQA_val_1217.json'
final_json_path = '/home/szx/project2/TAP/data/mid_json/VQAModel_out_TextVQA_val_1218.json'
# final_json_path = '/home/szx/project2/TAP/data/mid_json/VQAModel_out_STVQA_val_1218.json'

amazon_imdb_train_path = '/home/szx/project2/TAP/data/imdb_amazon_1201/textvqa/train_amazon_ocr.npy'
amazon_imdb_val_path = '/home/szx/project2/TAP/data/imdb_amazon_1201/textvqa/val_amazon_ocr.npy'
# amazon_imdb_test_path = '/home/szx/project/TAP/data/imdb_amazon/textvqa/test_amazon_ocr.npy'

# amazon_imdb_train_path = '/home/szx/project2/TAP/data/imdb/m4c_textvqa/imdb_train_ocr_en.npy'
# amazon_imdb_val_path = '/home/szx/project2/TAP/data/imdb/m4c_textvqa/imdb_val_ocr_en.npy'

amazon_imdb_train_stvqa_path = '/home/szx/project2/TAP/data/imdb_amazon_1201/stvqa/train_amazon_ocr.npy'
amazon_imdb_val_stvqa_path = '/home/szx/project2/TAP/data/imdb_amazon_1201/stvqa/val_amazon_ocr.npy'
# amazon_imdb_test_stvqa_path = '/home/szx/project/TAP/data/imdb_amazon/stvqa/test_amazon_ocr.npy'

# amazon_imdb_train_stvqa_path = '/home/szx/project2/TAP/data/original_dl/ST-VQA/m4c_stvqa/imdb_subtrain.npy'
# amazon_imdb_val_stvqa_path = '/home/szx/project2/TAP/data/original_dl/ST-VQA/m4c_stvqa/imdb_subval.npy'


# train_line = np.load(amazon_imdb_train_path, allow_pickle=True)
val_line = np.load(amazon_imdb_val_path, allow_pickle=True)
# train_stvqa = np.load(amazon_imdb_train_stvqa_path, allow_pickle=True)
val_stvqa = np.load(amazon_imdb_val_stvqa_path, allow_pickle=True)

val_json_file = []
with open(val_json_path,'r') as a:
    for line in a:
        # print(line)
        if line == '\n' or line == '':
            continue
        val_json_file.append(json.loads(line))

with open(final_json_path,'w') as final_out_file:
        final_out_file.write('\n')


ocr_tokens, questions, paths = get_prompts(val_line,val_stvqa,is_stvqa)

print(len(val_json_file))
print(len(ocr_tokens))
assert len(val_json_file) == len(ocr_tokens)

for val_jsons in val_json_file:
    for ocr_token, question, path in zip(ocr_tokens, questions, paths):
        # print(path)
        # print(question)
        # print(val_jsons)
        # exit()
        if is_stvqa:
            if 'data/stvqaimg/train/'+path == val_jsons['image_path'] and question == val_jsons['question'][0]:
                val_jsons['ocr_tokens'] = ocr_token
        else:
            if 'data/textvqaimg/'+path == val_jsons['image_path'] and question == val_jsons['question'][0]:
                val_jsons['ocr_tokens'] = ocr_token


# for val_jsons, ocr_token in zip(val_json_file,ocr_tokens):
#     # print(ocr_token)
#     val_jsons['ocr_tokens'] = ocr_token

with open(final_json_path,'a+') as final_out_file:
    for val_jsons in val_json_file:
        # print(val_jsons)
        json.dump(val_jsons,final_out_file)
        final_out_file.write('\n')
print('Finish! All ocr tokens loaded!')
# test_line = np.load(amazon_imdb_test_path, allow_pickle=True)
