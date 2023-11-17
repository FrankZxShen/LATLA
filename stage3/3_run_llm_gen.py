import re
import argparse
from tqdm import tqdm
import json
import torch
import pythia.utils.llama2_gen as llm_gen
from pythia.utils.llama2_gen import build_llama2

pre = ['is "','says :\n','says:\n','says: ','says','answer: ','is: \n','is:\n','is: ','was:\n','was: \n','was: ','was:','... ']
suf = ['" ','."','.\"','/','\"',', ','.\n','\n',', the','confidence score:',': 1.0',': 0.','. (1.0','. (0.',' (1.0',' (0.',' (','\nwith','.\n','".']
other = ['i apologize','ask your question','just an ai',]

def load_model(llm_path,lora_weights=None):
    print("Load Llama2")
    if lora_weights:
        model_llama2, tokenizer= build_llama2(llm_path, peft_model=lora_weights)
    else:
        model_llama2, tokenizer= build_llama2(llm_path)
    return model_llama2,tokenizer

def create_llm_prompt(samples, num_question_per_img=30):
        syn_question_queid = samples["questions"]
        syn_ans_queid = samples["answers"]
        Task_Prompt = ""
        num_question_generation = len(syn_question_queid) if len(syn_question_queid) <=num_question_per_img else num_question_per_img
        for idx in range(num_question_generation):
            Task_Prompt += "Question: "
            Task_Prompt += syn_question_queid[idx]
            Task_Prompt += "\n"
            Task_Prompt += "Candidate Answer: "
            Task_Prompt += syn_ans_queid[idx]
            Task_Prompt += "\n"
        
        samples["Task_Prompt"] = Task_Prompt
        # print(Task_Prompt)
        return Task_Prompt
# def get_pred_as(qa):
#     candidates = re.findall(r'(\w+):\s*1\.0',qa['candidates'])
#     if len(candidates) == 0:
#         return qa['pred_answer']
#     return candidates[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-llm', '--llama2', default='/DATACENTER1/szx/tap/data/models/llama-2-7b-hf/', help='name of llm model to use')
    parser.add_argument('-q', '--qagen', default='/DATACENTER1/szx/tap/data/mid_json/val_qa_gen_1113.json', help='path to qa file')
    parser.add_argument('-vqa', '--vqaout', default='/DATACENTER1/szx/tap/data/mid_json/VQAModel_out_TextVQA_val_1107.json', help='path to vqa model gen file')
    parser.add_argument('-vqa2', '--stvqaout', default='/DATACENTER1/szx/tap/data/mid_json/VQAModel_out_STVQA_val_1110.json', help='path to vqa model gen file')
    parser.add_argument('-llmo', '--llmout', default='/DATACENTER1/szx/tap/data/mid_json/val_llm_gen_1117_8bit.json', help='path to llm gen file')
    parser.add_argument('--lora', default='/home/szx/project/llama2-lora-fine-tuning/datasets/lora_weights/11-02/checkpoint-3000/', help='path to lora')
    args = parser.parse_args()
    all_qa_gen = []

    # args.lora = None

    with open(args.llmout,'w') as llm_out_file:
            llm_out_file.write('\n')
    if args.lora:
        model_llm, tokenizer = load_model(args.llama2,args.lora)
    else:
        model_llm, tokenizer = load_model(args.llama2)
    with open(args.qagen,'r') as qa_gen_file:
        for single_batch in qa_gen_file:
            if single_batch == '\n' or single_batch == {}:
                continue
            single_qa_gen = json.loads(single_batch)
            all_qa_gen.append(single_qa_gen)
    all_vqa_out = []
    with open(args.vqaout,'r') as vqa_out_file:
        for vqa_out in vqa_out_file:
            if vqa_out == '\n' or vqa_out == {}:
                continue
            single_vqa_out = json.loads(vqa_out)
            all_vqa_out.append(single_vqa_out)
    with open(args.stvqaout,'r') as vqa_out_file2:
        for vqa_out in vqa_out_file2:
            if vqa_out == '\n' or vqa_out == {}:
                continue
            single_vqa_out = json.loads(vqa_out)
            all_vqa_out.append(single_vqa_out)
    
    disable_tqdm=False
    for qa in tqdm(all_qa_gen, disable=disable_tqdm):
        context_prompt = qa["captions"][0][0]
        task_prompt = create_llm_prompt(qa)
        qa['pred_answer'] = None
        tmp_vqaans1 = None
        tmp_vqaans2 = None
        tmp_vqaans3 = None
        for vqa_ans in all_vqa_out:
            tg_path = ['data/textvqaimg/','data/stvqaimg/train/']
            index1 = vqa_ans['image_path'].find(tg_path[0])
            index2 = vqa_ans['image_path'].find(tg_path[1])
            if index1 != -1:
                # print(index1+len(tg_path[0]))
                vqa_ans['image_path'] = vqa_ans['image_path'][index1 + len(tg_path[0]):]
                tmp_vqaans1 = vqa_ans['image_path']
            elif index2 != -1:
                vqa_ans['image_path'] = vqa_ans['image_path'][index2 + len(tg_path[1]):]
                tmp_vqaans2 = vqa_ans['image_path']
            else:
                vqa_ans['image_path'] = vqa_ans['image_path']
                tmp_vqaans3 = vqa_ans['image_path']
            if qa['image_path'] == vqa_ans['image_path']:
                qa['pred_answer'] = vqa_ans['pred_ans']
                break

        # assert qa['pred_answer']
        # if not qa['pred_answer']:
        #     print("Error!!!!!!")
        #     print(qa['pred_answer'])
        #     print(tmp_vqaans1)
        #     print(tmp_vqaans2)
        #     print(tmp_vqaans3)
        #     exit()
        if len(qa['pred_answer']) < 1:
            qa['pred_answer'] = qa['answer']
        # qa['pred_answer'] = qa['answer']
        LLMPrompt = (
            "Context: "
            + context_prompt
            + "\n"
            + task_prompt
            + "======\n"
            + "Question: "
            + qa["question"]
            + "\n"
            + "Predict Answer: "
            + qa['pred_answer']
            + "\n"
            + "Based on the Questions and Answers above, answer the following question using a single word or phrase.\n"
            + "Question: "
            + qa["question"]
            + "\n"
            + "Answer: "
        )
        pre_prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human\'s questions. \
The assistant should answer the following question based on the above contexts and answer candidates. The true answer may not be included in the candidate. \
The assistant will not respond to anything other than that word. For example, the question is \'what is the volume 1 about?\', you would reply with only \'the family of dashwood\'. If don't know, the reply will be \'unanswerable\'."
        LLM_outputs = llm_gen.generate(
            model=model_llm,
            tokenizer=tokenizer,
            user_prompt=LLMPrompt,
            pre_prompt=pre_prompt,
        )
        LLM_outputs = LLM_outputs.lower()
        print("LLM_origin:",LLM_outputs)
        # writer.write("Question:")
        # writer.write(entry["questions"])
        # writer.write("LLM:")
        # writer.write(LLM_outputs)
        llm_out = {}
        llm_out['image_path'] = qa['image_path']
        llm_out['pred'] = qa['pred_answer']
        llm_out['llm_origin'] = LLM_outputs
        
        for pre_context in pre:
            if LLM_outputs.find(pre_context)!= -1:
                LLM_outputs = LLM_outputs[LLM_outputs.find(pre_context)+len(pre_context):]
        for suf_context in suf:
            if LLM_outputs.find(suf_context)!= -1:
                LLM_outputs = LLM_outputs[:LLM_outputs.find(suf_context)]
        for other_context in other:
            if LLM_outputs.find(other_context)!= -1:
                LLM_outputs = 'unanswerable'
        
        llm_out['llm_out'] = LLM_outputs
        print("pred:",qa['pred_answer'])
        print("LLM:",LLM_outputs)
        print("=============================")
        with open(args.llmout,'a+') as llm_out_file:
            json.dump(llm_out,llm_out_file)
            llm_out_file.write('\n')

if __name__ == "__main__":
    main()
