# Part of the code is adapted from https://github.com/allenai/open-instruct
import json
import json
from transformers import AutoTokenizer, LlamaForCausalLM
import torch
from tqdm import tqdm
from transformers import StoppingCriteria

data_name = 'casa_llama2.json'
data = json.load(open(data_name, 'r'))
model_name = 'Llama-2-7b-chat-hf'

generate_instruction = 'Generate 3 detailed contexts. Each context is consistent with both the premise and the conclusion. Each context is in one line.'
generate_instruction_context = 'Generate 3 detailed contexts. Each context contains "{cur_context}" Each context is consistent with both the premise and the conclusion. Each context is in one line.'

if 'Llama' in model_name:
    generate_prompt_format = "### Instruction:\n{instruction}\n\n### Input:\nPremise: {neg_premise}\nConclusion: {neg_conclusion}\n\n### Response:"
elif 'Tulu' in model_name:
    generate_prompt_format = "<|user|>\n### Instruction:\n{instruction}\n\n### Input:\nPremise: {neg_premise}\nConclusion: {neg_conclusion}\n\n### Response:\n<|assistant|>"

prompts = []
idxs = []
for idxi, i in tqdm(enumerate(data)):
    if not 'neg_premise' in i:
        continue
    for j in range(len(i['premise'])):
        cur_neg_premise = i['neg_premise'][j].strip()
        cur_premise = i['premise'][j].strip()
        cur_context = ". ".join([x for k,x in enumerate(i['premise']) if k!=j]).strip()
        # generate contexts with neg_premise, neg_conclusion
        if len(cur_context) > 0:
            generate_context_instance = generate_prompt_format.format(instruction=generate_instruction_context.format(cur_context = cur_context), neg_premise = cur_neg_premise, neg_conclusion = i['neg_conclusion'])
        else:
            generate_context_instance = generate_prompt_format.format(instruction=generate_instruction, neg_premise = cur_neg_premise, neg_conclusion = i['neg_conclusion'])
        prompts.append(generate_context_instance)
        idxs.append([idxi, j])
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
            print(batch_generations[0].strip())
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

for idxi, i in enumerate(responses):
    i = i.split('Response:')[-1].strip()
    response_reformat = []
    cur = ""
    for k in i.split('\n'):
        if len(k) == 0:
            continue
        if len(cur) >= 8 and cur[-1] == ':':
            cur += k
        else:
            response_reformat.append(cur.strip())
            cur = k
    response_reformat.append(cur.strip())

    cur_neg_contexts = []
    for k in response_reformat:
        if 'Context' in k and ':' in k:
            cur_neg_contexts.append(k.split(':')[1].strip())
        elif len(k)>2 and k[0]>='1' and k[0]<='9' and k[1]=='.':
            cur_neg_contexts.append(k[2:].strip())
    
    if not 'neg_context' in data[idxs[idxi][0]]:
        data[idxs[idxi][0]]['neg_context'] = [cur_neg_contexts]
    else:
        data[idxs[idxi][0]]['neg_context'].append(cur_neg_contexts)
        
with open(data_name, 'w') as f:
    f.write(json.dumps(data, indent=2))