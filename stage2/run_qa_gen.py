from PIL import Image
import argparse
from tqdm import tqdm
from typing import List, Optional
import spacy
import torch
import gc
from transformers import (
    AutoProcessor, AutoModelForCausalLM, BlipForConditionalGeneration, T5ForConditionalGeneration, T5Tokenizer, 
    logging
)
import json

open_pos = ["NOUN", "VERB", "ADJ", "ADV", "NUM"]
class QAGeneration:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.device = torch.device("cuda", 0)

    def load_models(self,t5_large_lm_adapt_path,T5_large_QG_path):
        print("Load T5 for QA generation")
        self.tokenizer_t5 = T5Tokenizer.from_pretrained(
            t5_large_lm_adapt_path
        )
        self.model_t5 = T5ForConditionalGeneration.from_pretrained(
            t5_large_lm_adapt_path
        ).to(self.device)
        cache_file = T5_large_QG_path
        checkpoint = torch.load(cache_file, map_location="cpu")
        state_dict = checkpoint["model"]
        self.model_t5.load_state_dict(state_dict)
        return self.model_t5,self.tokenizer_t5
    def answer_extraction(self, caption, num_question_generation=20):
            cap_use = ""
            # print(caption)
            caption = caption
            ans_to_cap_dict = {}
            answers = []
            for cap_idx, cap in enumerate(caption):
                # print(cap)
                cap_use += cap
                cap = cap.strip().strip(".")
                # print(cap)
                cap = self.nlp(cap)
                for token in cap:  # Noun /Verb/Adj//NUM
                    if token.pos_ in open_pos:
                        if token.text.lower() not in ans_to_cap_dict:
                            ans_to_cap_dict[token.text.lower()] = [cap_idx]
                        else:
                            if cap_idx not in ans_to_cap_dict[token.text.lower()]:
                                ans_to_cap_dict[token.text.lower()].append(cap_idx)
                        answers.append(token.text)
                for ent in cap.ents:

                    if ent.text not in answers:
                        if ent.text.lower() not in ans_to_cap_dict:
                            ans_to_cap_dict[ent.text.lower()] = [cap_idx]
                        else:
                            if cap_idx not in ans_to_cap_dict[ent.text.lower()]:
                                ans_to_cap_dict[ent.text.lower()].append(cap_idx)
                        answers.append(ent.text)
                for chunk in cap.noun_chunks:
                    if len(chunk.text.split()) < 4:
                        if chunk.text.lower() not in ans_to_cap_dict:
                            ans_to_cap_dict[chunk.text.lower()] = [cap_idx]
                        else:
                            if cap_idx not in ans_to_cap_dict[chunk.text.lower()]:
                                ans_to_cap_dict[chunk.text.lower()].append(cap_idx)
                        #                 print(chunk.text)
                        answers.append(chunk.text)
            answers = sorted(answers, key=answers.count, reverse=True)
            real_answers = []
            for i in answers:
                i = i + "."
                if i not in real_answers:
                    real_answers.append(i)

            contexts_for_question_generation = []
            answers = []
            for ans in real_answers[
                :num_question_generation
            ]:  # Generate questions for 15 answers with max frequencies.
                contexts_for_question_generation.append(
                    "answer: %s  context: %s." % (ans, cap_use)
                )
                answers.append(ans)
            contexts_for_question_generation.append(
                "answer: %s  context: %s." % ("yes.", cap_use)
            )
            answers.append("yes.")
            return contexts_for_question_generation, answers, ans_to_cap_dict
        
    # QA gen
    def forward_qa_generation(self, samples, question_generation_tokenizer, question_generation_model):
        caption = samples["captions"][0]
        (
            contexts_for_question_generation,
            answers,
            ans_to_cap_dict,
        ) = self.answer_extraction(caption)
        # print("*******************************")
        # print(contexts_for_question_generation)
        # print("*******************************")
        inputs = question_generation_tokenizer(
            contexts_for_question_generation,
            padding="longest",
            truncation=True,
            max_length=2048,
            return_tensors="pt",
        ).to(question_generation_model.device)
        question_size = inputs.input_ids.shape[0]
        cur_b = 0
        true_input_size = 10
        outputs_list = []
        question_generation_model = question_generation_model.to(inputs.input_ids.device)
        # print("OK")
        while cur_b < question_size:
            outputs = question_generation_model.generate(
                input_ids=inputs.input_ids[cur_b : cur_b + true_input_size],
                attention_mask=inputs.attention_mask[cur_b : cur_b + true_input_size],
                num_beams=3,
                max_length=20,
            )
            questions = question_generation_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            outputs_list += questions
            cur_b += true_input_size
        questions = outputs_list
        samples["questions"] = questions
        samples["answers"] = answers
        samples["ans_to_cap_dict"] = ans_to_cap_dict
        # results.append({"question_id": ques_id, "question":questions,"answer":answers})
        return samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-lm', '--t5lm', default='/DATACENTER1/szx/tap/data/models/t5-large-lm-adapt/', help='name of t5lm model to use')
    parser.add_argument('-qa', '--t5qa', default='/DATACENTER1/szx/tap/data/models/T5-large-QG/T5_large_QG.pth', help='name of t5QG model to use')
    parser.add_argument('-c', '--caption', default='/DATACENTER1/szx/tap/data/mid_json/texvqa_caption_gen.json', help='path to caption file')
    parser.add_argument('-q', '--qagen', default='/DATACENTER1/szx/tap/data/mid_json/texvqa_qa_gen.json', help='path to qa file')

    args = parser.parse_args()
    QA = QAGeneration()
    t5_lm,t5_tokenizer= QA.load_models(t5_large_lm_adapt_path=args.t5lm,T5_large_QG_path=args.t5qa)
    with torch.no_grad():
        cap = {"captions":[[[]]]}
        all_captions = []
        with open(args.qagen,'w') as qa_gen_file:
            qa_gen_file.write('\n')
        with open(args.caption,'r') as caption_file:
            for img_captions in caption_file:
                img_caption = json.loads(img_captions)
                all_captions.append(img_caption)
                # print(img_caption)
        disable_tqdm = False
        for img_caption in tqdm(all_captions, disable=disable_tqdm):
            if img_caption == {}:
                continue
            # print("img_name1:",image_name[0])
            # print("img_name2:",list(single_img.keys())[0])
            cap['captions'][0][0] = img_caption['captions']
            single_qa_gen = QA.forward_qa_generation(cap,t5_tokenizer,t5_lm)
            single_qa_gen['image_path'] = img_caption['image_path']
            single_qa_gen['question'] = img_caption['question']
            single_qa_gen['candidates'] = img_caption['candidates']
            single_qa_gen['pred_answer'] =img_caption['pred_answer']
            print(single_qa_gen)
            # exit()
            # if list(img_caption.keys())[0] == image_name[0]:
            #     cap["captions"][0].append(single_img.get(image_name[0]))
            #     break
            with open(args.qagen,'a+') as qa_gen_file:
                json.dump(single_qa_gen,qa_gen_file)
                qa_gen_file.write('\n')
   


if __name__ == "__main__":
    main()
