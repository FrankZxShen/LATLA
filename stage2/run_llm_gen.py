import re
import argparse
from tqdm import tqdm
import json
import torch
import pythia.utils.llama2_gen as llm_gen
from pythia.utils.llama2_gen import build_llama2

pre = ['is "','says :\n','says:\n','says: ','says','is: \n','is:\n','is: ','was:\n','was: \n','was: ','was:','... ']
suf = ['" ','."','.\"','/','\"',', ','.\n','\n',', the','confidence score:',': 1.0',': 0.','. (1.0','. (0.',' (1.0',' (0.',' (','\nwith','.\n','".']
other = ['i apologize','ask your question','just an ai',]

def load_model(llm_path):
    print("Load Llama2")
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
            Task_Prompt += "Answer: "
            Task_Prompt += syn_ans_queid[idx]
            Task_Prompt += "\n"
        
        samples["Task_Prompt"] = Task_Prompt
        # print(Task_Prompt)
        return Task_Prompt
def get_pred_as(qa):
    candidates = re.findall(r'(\w+):\s*1\.0',qa['candidates'])
    if len(candidates) == 0:
        return qa['pred_answer']
    return candidates[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-llm', '--llama2', default='/DATACENTER1/szx/tap/data/models/llama-2-7b-hf/', help='name of llm model to use')
    parser.add_argument('-q', '--qagen', default='/DATACENTER1/szx/tap/data/mid_json/texvqa_vlm_qa_gen.json', help='path to qa file')
    parser.add_argument('-lo', '--llmout', default='/DATACENTER1/szx/tap/data/mid_json/texvqa_vlm_llm_gen.json', help='path to qa file')
    args = parser.parse_args()
    all_qa_gen = []
    with open(args.llmout,'w') as llm_out_file:
            llm_out_file.write('\n')
    model_llm, tokenizer = load_model(args.llama2)
    with open(args.qagen,'r') as qa_gen_file:
        for single_batch in qa_gen_file:
            if single_batch == '\n' or single_batch == {}:
                continue
            single_qa_gen = json.loads(single_batch)
            all_qa_gen.append(single_qa_gen)
    disable_tqdm=False
    for qa in tqdm(all_qa_gen, disable=disable_tqdm):
        context_prompt = qa["captions"][0][0]
        task_prompt = create_llm_prompt(qa)
        qa['pred_answer'] = get_pred_as(qa)
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
        LLM_outputs = llm_gen.generate(
            model=model_llm,
            tokenizer=tokenizer,
            user_prompt=LLMPrompt,
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
