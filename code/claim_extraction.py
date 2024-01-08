# Part of the code is adapted from https://github.com/allenai/open-instruct
import json
from transformers import AutoTokenizer, LlamaForCausalLM
import torch
from tqdm import tqdm
from transformers import StoppingCriteria
import re
from negate import Negator

with open('../data/Bigbench-LFD.json', 'r') as f:
    data = json.load(f)

data_name = 'casa_llama2.json'
model_name = 'Llama-2-7b-chat-hf'

extraction_instruction = 'Determine which part of the text is the conclusion.\nOutput the number of the conclusion part first, and give an explanation.\nFormat:\nConclusion: [number]\nExplanation: ...'
if 'Llama' in model_name:
    extraction_prompt_format = "### Instruction:\n{instruction}\n\n### Input:\n{text}\n\n### Response:"
elif 'Tulu' in model_name:
    extraction_prompt_format = "<|user|>\n### Instruction:\n{instruction}\n\n### Input:\n{text}\n\n### Response:\n<|assistant|>"
    
# argument segmentation
prompts = []
idxs = []
splitted_text = []
for idxi, i in tqdm(enumerate(data)):
    text = i['text']+'\nChoices:\n'
    cur_splitted = []
    red = [j for j in re.split('; |, |\. |\? |- ', i['text']) if len(j.strip()) >= 10]
    if len(red) == 1:
        for j in red:
            tmp = re.split(' because | so | if | and | but | otherwise | or |  Because | So | If | And | But | Otherwise | Or ', ' '+j.strip()+' ')
            for k in tmp:
                if len(k.strip()) < 10:
                    continue
                text += str(number) + '. ' + k.strip() + '\n'
                cur_splitted.append(k.strip())
    else:
        for j in red:
            text += str(number) + '. ' + j.strip() + '\n'
            cur_splitted.append(j.strip())

    extraction_instance = extraction_prompt_format.format(instruction=extraction_instruction, text=text.strip())
    prompts.append(extraction_instance)
    idxs.append(idxi)
    splitted_text.append(cur_splitted)

print(len(prompts))
        
model = LlamaForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"

device = torch.device("cuda") 
model.to(device)

class KeyWordsCriteria(StoppingCriteria):
    def __init__(self, stop_id_sequences):
        assert isinstance(stop_id_sequences[0], list), "stop_id_sequences should be a list of list of ids"
        self.stop_sequences = stop_id_sequences

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        sequences_should_be_stopped = []
        for i in range(input_ids.shape[0]):
            sequence_should_be_stopped = False
            for stop_sequence in self.stop_sequences:
                if input_ids[i][-len(stop_sequence):].tolist() == stop_sequence:
                    sequence_should_be_stopped = True
                    break
            sequences_should_be_stopped.append(sequence_should_be_stopped)
        return all(sequences_should_be_stopped)

@torch.no_grad()
def generate_completions(model, tokenizer, prompts, batch_size=1, stop_id_sequences=None, **generation_kwargs):
    generations = []

    num_return_sequences = generation_kwargs.get("num_return_sequences", 1)
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i+batch_size]
        tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=False)
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask

        if model.device.type == "cuda":
            batch_input_ids = batch_input_ids.cuda()
            attention_mask = attention_mask.cuda()

        try:
            batch_outputs = model.generate(
                input_ids=batch_input_ids,
                attention_mask=attention_mask,
                stopping_criteria=[KeyWordsCriteria(stop_id_sequences)] if stop_id_sequences else None,
                **generation_kwargs
            )
        
            # the stopping criteria is applied at batch level, so if other examples are not stopped, the entire batch will continue to generate.
            # so some outputs still have the stop sequence, which we need to remove.
            if stop_id_sequences:
                for output_idx in range(batch_outputs.shape[0]):
                    for token_idx in range(batch_input_ids.shape[1], batch_outputs.shape[1]):
                        if any(batch_outputs[output_idx, token_idx: token_idx+len(stop_sequence)].tolist() == stop_sequence for stop_sequence in stop_id_sequences):
                            batch_outputs[output_idx, token_idx:] = tokenizer.pad_token_id
                            break

            # remove the prompt from the output
            # we need to re-encode the prompt because we need to make sure the special tokens are treated the same way as in the outputs.
            # we changed our previous way of truncating the output token ids dicrectly because some tokenizer (e.g., llama) won't add space token before the first token.
            # space is important for some tasks (e.g., code completion).
            batch_outputs = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
            batch_prompts = tokenizer.batch_decode(batch_input_ids, skip_special_tokens=True)
            # duplicate the prompts to match the number of return sequences
            batch_prompts = [prompt for prompt in batch_prompts for _ in range(num_return_sequences)]
            batch_generations = [
                output[len(prompt):] for prompt, output in zip(batch_prompts, batch_outputs)
            ]
        except Exception as e:
            print("Error when generating completions for batch:")
            print(batch_prompts)
            print("Error message:")
            print(e)
            print("Use empty string as the completion.")
            batch_generations = [""] * len(batch_prompts) * num_return_sequences

        generations += batch_generations

    assert len(generations) == len(prompts) * num_return_sequences, "number of generations should be equal to number of prompts * num_return_sequences"
    return generations


batch_size = 8
responses = generate_completions(model, tokenizer, prompts, batch_size=batch_size, max_length=512)

negator = Negator()

for idxi, i in enumerate(responses):
    i = i.split('Response:')[-1].strip()
    cur_splitted = splitted_text[idxi]
    for j in i.split('\n'):
        if 'Conclusion:' in j or (j[0]>='1' and j[0]<='9'):
            if 'Conclusion:' in j:
                conclusion_num = j.split('Conclusion:')[1].strip()[0]
            else:
                conclusion_num = j[0]
            try:
                conclusion = cur_splitted[int(conclusion_num)-1]
            except:
                conclusion = cur_splitted[-1]
            premise = [k for k in cur_splitted if k != conclusion]
            data[idxs[idxi]].update({'premise': premise, 'conclusion': conclusion})

            neg_premise = []
            for j in premise:
                neg_premise.append(negator.negate_sentence(j))
            data[idxs[idxi]].update({'neg_premise': neg_premise})
        
            neg_conclusion = negator.negate_sentence(conclusion)
            data[idxs[idxi]].update({'neg_conclusion': neg_conclusion})

            break

with open(data_name, 'w') as f:
    f.write(json.dumps(data, indent=2))