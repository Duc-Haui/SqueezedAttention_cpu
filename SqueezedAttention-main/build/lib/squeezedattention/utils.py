import os

# This is the customized building prompt for chat models
def build_chat(prompt, model_name, noquery=False):
    if noquery:
        if "LWM" in model_name:
            prompt = f"You are a helpful assistant. USER: {prompt}"
        elif "longchat" in model_name:
            from fastchat.model import get_conversation_template
            conv = get_conversation_template("vicuna")
            conv.append_message(conv.roles[0], prompt)
            prompt = conv.get_prompt()

    else:
        if "LWM" in model_name:
            prompt = f"You are a helpful assistant. USER: {prompt} ASSISTANT: "
        elif "longchat" in model_name:
            from fastchat.model import get_conversation_template
            conv = get_conversation_template("vicuna")
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
    return prompt

def truncate_fn(prompt, prompt_noquery, tokenizer, max_length, dataset, device):
    # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
    tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    tokenized_prompt_noquery = tokenizer(prompt_noquery, truncation=False, return_tensors="pt").input_ids[0]

    # truncate based on length of prompt with query
    len_tokenized_prompt = len(tokenized_prompt)
    if len(tokenized_prompt) > max_length:
        half = int(max_length/2)

        # compute num tokens removed and subtract from sp_len
        tokens_removed = len(tokenized_prompt) - 2*half

        prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
    else:
        tokens_removed = 0

    # incorporate chat template for shared prefix length
    if dataset not in ["trec", "triviaqa", "samsum", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
        prompt = build_chat(prompt, model_name)
        prompt_noquery = build_chat(prompt_noquery, model_name, noquery=True)

    # compute shared prefix length
    input_ids_prompt_only = tokenizer(prompt_noquery, truncation=False, return_tensors="pt").input_ids.to(device)
    shared_prefix_length = input_ids_prompt_only.shape[1]

    return prompt, shared_prefix_length - tokens_removed
