import os
import json
import pandas as pd
import numpy as np
from data_utils import extract_gold, extract_results, parse_bleu_scores, parse_use_scores
from tenacity import retry, stop_after_attempt, wait_fixed
from tqdm import tqdm
from lmwrapper.structs import LmPrompt, LmChatTurn
from lmwrapper.openai_wrapper import get_open_ai_lm, OpenAiModelNames, OpenAIPredictor
import json
import tiktoken
import openai
import re

likert_scale = {1: "Strongly Disagree", 2: "Disagree", 3: "Agree", 4: "Strongly Agree"}

def diverse_example(target_score, moe=0.1):
    haque_et_al_path = './data/haque_et_al/final_megafile.csv'
    df = pd.read_csv(haque_et_al_path)

    # Ensures that in a 4-shot setting, we represent all possible ratings
    i = 0
    fids = df['function_id'].unique()
    candidates = {} # maps fid to diff btwn sim and adq for all summaries with sim = target score 
    for i in range(len(fids)):
        fid = fids[i]
        df_baseline = df[(df['function_id'] == fid) & (df['source'] == 'baseline')]
        # was rounded for human ratings
        rounded_similarity_rating = np.round(np.mean(df_baseline['similarity']))
        if rounded_similarity_rating == target_score:
            df_reference = df[(df['function_id'] == fid) & (df['source'] == 'reference')]
            #similarity_rating = np.mean(df_baseline['similarity'])
            adequate_rating = np.mean(df_reference['adequate'])
            #candidates[fid] = np.abs(similarity_rating-adequate_rating)
            candidates[fid] = adequate_rating
    fid = max(candidates, key=candidates.get) # Reference should be good for similarity to be valid
    print(f"Best adequacy value for {target_score}:", candidates[fid])
    df_baseline = df[(df['function_id'] == fid) & (df['source'] == 'baseline')]
    df_reference = df[(df['function_id'] == fid) & (df['source'] == 'reference')]
    gold_summary = df_reference['text'].iloc[0]
    generated_summary = df_baseline['text'].iloc[0]
    
    fid_to_text ='./data/haque_et_al/functions.json'
    with open(fid_to_text, 'r') as f:
        fid_to_text = json.load(f)
    function = fid_to_text[str(fid)]

    return function, generated_summary, gold_summary, target_score

def similarity_query(method, summary):
    return ("Given the following code method and summary, express your agreement from 1 to 4, "
            "where 1 indicates no similarity between the given summary and what the method's developer would write while 4 indicates perfect similarity to what the method's developer would write. "
            "The method's developer writes maximum one line summaries which only contain information important to understanding the method’s functionality.\n"
            f"Method: {method}\n"
            f"Summary: {summary}\n"
            "Provide your reasoning in one sentence, then give a final score. Strictly follow the following format. Reasoning: {Concise one sentence explanation} Score: {your score}.")

def similarity_example(target_score, measure='adequate'):
    method, generated_summary, gold_summary, example_human_rating = diverse_example(target_score, measure)
    prompt = {'assistant': '', 'user': ''}
    prompt['user'] += similarity_query(method, gold_summary)
    if target_score == 1:
        reasoning = "Reasoning: The summary is missing all of the information the method's developer would include."
    elif target_score == 2:
        reasoning = "Reasoning: The summary is missing the majority of the information the method's developer would include."
    elif target_score == 3:
        reasoning = "Reasoning: The summary captures the majority of the information the developer would include while remaining concise."
    elif target_score == 4:
        reasoning = "Reasoning: The summary captures all of the information the developer would include while remaining concise."
    prompt['assistant'] += f"{reasoning} Score: " + str(int(example_human_rating))
    return prompt

def true_or_false_query(method, summary):
    return (
        f"Method: {method}\n"
        f"Summary: {summary}\n"
        f"The method's developer writes maximum one line summaries that only contain information important to understanding the method’s functionality.\n True or False: The given summary is similar to what the method's developer would write. Only reply with a single word \"True\" or \"False\":"
        )

def generate_prompts(num_shots, language, model_files, num_prompts, true_or_false=False):
    results_dir = f"./data/{language}/model_outputs/"
    gold_file_path = f"./data/{language}/prompting_data/gold.txt"
    test_data_path = f"./data/{language}/prompting_data/test.jsonl"
    sentence_bert_dir = f"./data/{language}/metrics_results/sbert/"
    bleu1_dir = f"./data/{language}/metrics_results/bleu-1/"
    infersent_dir = f"./data/{language}/metrics_results/infersent/"
    bertscore_dir = f"./data/{language}/metrics_results/bert-score-recall/"
    rogue1p_dir = f"./data/{language}/metrics_results/rouge-1-p/"
    rogue4r_dir = f"./data/{language}/metrics_results/rouge-4-r/"
    roguewr_dir = f"./data/{language}/metrics_results/rouge-w-r/"

    gold_summaries = extract_gold(gold_file_path)

    functions = []
    with open(test_data_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            prompt = data['code']
            functions.append(prompt)
    
    examples = []
    queries = []
    for model_filename in model_files:
        results_path = os.path.join(results_dir, model_filename)
        sentence_bert_path = os.path.join(sentence_bert_dir, model_filename)
        bleu1_path = os.path.join(bleu1_dir, model_filename)
        infersent_path = os.path.join(infersent_dir, model_filename)
        bertscore_path = os.path.join(bertscore_dir, model_filename)
        rogue1p_path = os.path.join(rogue1p_dir, model_filename)
        rogue4r_path = os.path.join(rogue4r_dir, model_filename)
        roguewr_path = os.path.join(roguewr_dir, model_filename)

        metrics_results, invalid_results = extract_results(results_path)

        gold_summaries_for_results = [gold_summaries[i] for i in metrics_results.keys()]
        generated_summaries = [result['generated_summary'] for result in metrics_results.values()]

        sentence_bert_scores = parse_bleu_scores(sentence_bert_path)  
        bleu1_scores = parse_bleu_scores(bleu1_path)  
        infersent_scores = parse_bleu_scores(infersent_path)  
        bertscore_scores = parse_bleu_scores(bertscore_path)  
        rogue1p_scores = parse_bleu_scores(rogue1p_path)  
        rogue4r_scores = parse_bleu_scores(rogue4r_path)  
        roguewr_scores = parse_bleu_scores(roguewr_path)  

        # Each example has human ratings between 1 and num_shots. Max num_shots = 4
        for i in range(num_shots):
            # i + 1 because we want to start an example with human rating 1 but 0-indexing
            example = similarity_example(i + 1)
            examples.append(example)

        for i, (function, gold_summary, generated_summary, sentence_bert_score, bleu1_score, bertscore_score, infersent_score, rogue1p_score, roguewr_score, rogue4r_score) in\
              enumerate(zip(functions, gold_summaries_for_results, generated_summaries, sentence_bert_scores, bleu1_scores, bertscore_scores, infersent_scores, rogue1p_scores, roguewr_scores, rogue4r_scores)):
            query_info = {}
            if true_or_false:
                query_info['query'] = true_or_false_query(function, generated_summary)
            else:
                query_info['query'] = similarity_query(function, generated_summary)
            query_info['code'] = function
            query_info['Model Generated Summary'] = generated_summary
            query_info['Human Gold Summary'] = gold_summary
            query_info['SentenceBERT_CS'] = sentence_bert_score
            query_info['Model'] = model_filename[:-4]
            query_info['BLEU-1'] = bleu1_score
            query_info['BERTScore-R'] = bertscore_score
            query_info['InferSent_CS'] = infersent_score
            query_info['ROUGE-1-P'] = rogue1p_score
            query_info['ROUGE-W-R'] = roguewr_score
            query_info['ROUGE-4-R'] = rogue4r_score
            queries.append(query_info)
            if i == (num_prompts - 1): break

    return examples, queries

def count_tokens(str, model):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(str)
    return len(tokens)

import random
from collections import Counter

def majority_vote(values):
    if values == []:
        return None
    value_counts = Counter(values)

    max_count = max(value_counts.values())
    
    candidates = [value for value, count in value_counts.items() if count == max_count]
    # Handle ties
    if len(candidates) == 1:
        # No tie, return the single candidate
        return candidates[0]
    else:
        # Tie exists, choose randomly among the candidates
        return random.choice(candidates)

def messages_to_string(messages):
    # messages is a list of dictionaries with content entry. Concatenate all
    # content entries and separate with a new line
    return '\n'.join([message['content'] for message in messages])

def prompt_code_llama(model, messages):
    # low T
    url = "https://api.together.xyz/v1/completions"
    payload = {
        "model": model,
        "prompt": "[INST]" + messages_to_string(messages) + "[/INST]",
        "max_tokens": 100,
        "stop": [],
        "temperature": 0,
        "top_p": 1,
        "n": 1,
        "logprobs": True,
        "repetition_penalty": 1
    }

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": "REPLACE"
    }

    import requests
    api_response = requests.post(url, json=payload, headers=headers)
    chat_completion = json.loads(api_response.text)
    response = chat_completion["choices"][0]["text"]
    match = re.search(r'Score: (\d+)', response)
    if match:
        rating = match.group(1)  # group(1) to get the first capturing group
        rating_logprobs = None
    else:
        rating = None
        rating_logprobs = None
    return rating, [{"pred": chat_completion, "completion": response, "rating_logprobs": rating_logprobs}]

def construct_messages(prompt):
    return [{"role": message.role, "content": message.content} for message in prompt]

def call_openai_chat_completion(model, messages, T, num_completions, reflective_logit):
    chat_completion = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=T,
        logprobs=True,
        top_logprobs=20,
        n=num_completions,
        max_tokens=200,
        stop=["<|im_end|>"]
    )
    return process_chat_completion_response(chat_completion, reflective_logit)

def process_chat_completion_response(chat_completion, reflective_logit):
    response = chat_completion.choices[0].message.content
    if reflective_logit:
        return process_reflective_logit_response(chat_completion, response)
    else:
        return process_standard_response(chat_completion, response)

def process_reflective_logit_response(chat_completion, response):
    rating = response
    top_logprobs = chat_completion.choices[0]["logprobs"]["content"][0]["top_logprobs"]
    true_found = False
    for top_logprob in top_logprobs:
        if top_logprob["token"] == "True": 
            rating_logprobs = [top_logprob["logprob"]]
            true_found = True
    if not true_found: 
        rating_logprobs = None
        print("No True found. top_logprobs=", top_logprobs)
    return rating, [{"pred": chat_completion, "completion": response, "rating_logprobs": rating_logprobs}]

def process_standard_response(chat_completion, response):
    match = re.search(r'Score: (\d+)', response)
    if match:
        rating = match.group(1) 
        rating_logprobs = None
    else:
        rating = None
        rating_logprobs = None
    return rating, [{"pred": chat_completion, "completion": response, "rating_logprobs": rating_logprobs}]

def setup_together_api():
    openai.api_key = os.environ.get("TOGETHER_API_KEY")
    openai.api_base = "https://api.together.xyz/v1"

def setup_openai_api():
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    openai.api_base = "https://api.openai.com/v1"

def prompt_lm(model, prompt: list[LmChatTurn], T, together=False, reflective_logit=False):
    messages = construct_messages(prompt)
    num_completions = 1 if T == 0 else 5

    if together:
        setup_together_api()
    else:
        setup_openai_api()

    if T == 0:
        if model == "codellama/CodeLlama-70b-hf":
            # completion model is handled separately
            rating, response_info = prompt_code_llama(model, messages)
        else:
            rating, response_info = call_openai_chat_completion(model, messages, T, num_completions, reflective_logit)
    else:
        if model == "codellama/CodeLlama-70b-hf":
            rating, response_info = prompt_code_llama_with_temperature(model, messages, T)
        else:
            rating, response_info = openai_chat_completion_with_temperature(model, messages, T, num_completions)

    if reflective_logit and rating not in ["True", "False"]:
        handle_reflective_logit_error(messages, rating)
    return rating, response_info

def prompt_code_llama_with_temperature(model, messages, T):
    url = "https://api.together.xyz/v1/completions"
    choices = []
    for i in range(1, 6):
        payload = {
            "model": model,
            "prompt": "[INST]" + messages_to_string(messages) + "[/INST]",
            "max_tokens": 100,
            "stop": [],
            "temperature": T,
            "top_p": 1,
            "n": 1,
            "logprobs": True,
            "repetition_penalty": 1
        }

        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": "Bearer 9c64e32d4da7da57ad79b203b28b6582c4467dca72b110793289615ad83bb751"
        }

        import requests
        api_response = requests.post(url, json=payload, headers=headers)
        chat_completion = json.loads(api_response.text)
        choices.append(chat_completion["choices"][0])

    ratings = []
    rating_to_samples = {}
    for i, choice in enumerate(choices):
        response = choice["text"]
        match = re.search(r'Score: (\d+)', response)
        if match:
            rating = match.group(1)  # group(1) to get the first capturing group
            rating_logprobs = None
            ratings.append(rating)
            if rating not in rating_to_samples:
                rating_to_samples[rating] = []
            rating_to_samples[rating].append({"rating_logprobs": rating_logprobs, "completion": response, "pred": chat_completion})
    majority_rating = majority_vote(ratings)
    rating_logprobs = None
    return majority_rating, rating_to_samples[majority_rating]

def openai_chat_completion_with_temperature(model, messages, T, num_completions):
    #num_completions paramter behaving weirdly so workaround
    choices = []
    for i in range(1, 6):
        chat_completion = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=T,
            logprobs=True,
            top_logprobs=20,
            n=1,
            max_tokens=200,
            stop=["<|im_end|>"]
        )
        choices.append(chat_completion["choices"][0])
    ratings = []
    rating_to_samples = {}
    for i, choice in enumerate(choices):
        response = choice.message.content
        rating = response.split(': ')[-1][0]
        ratings.append(rating)

        num_rating_tokens = count_tokens(rating, OpenAiModelNames.gpt_3_5_turbo)
        rating_logprobs = choice.logprobs.token_logprobs[-num_rating_tokens:]
        if rating not in rating_to_samples:
            rating_to_samples[rating] = []
        rating_to_samples[rating].append({"rating_logprobs": rating_logprobs, "completion": response, "pred": chat_completion})
    majority_rating = majority_vote(ratings)
    return majority_rating, rating_to_samples[majority_rating]

def handle_reflective_logit_error(messages, rating):
    print(messages_to_string(messages))
    print(rating)
    raise ValueError(f"Not True or False response: {rating}")

if __name__ == '__main__':  
    models = [OpenAiModelNames.gpt_3_5_turbo, "deepseek-ai/deepseek-coder-33b-instruct", "codellama/CodeLlama-70b-hf"] 
    model_names = ["gpt-3.5-turbo", "deepseek-coder-33b-instruct", "CodeLlama-70b-hf"]
    model_files = ["gpt-3.5-turbo_ASAP.txt", "deepseek-coder-33b-instruct_ASAP.txt", "CodeLlama-70b-hf_ASAP.txt"]
    reflective_logit = False
    num_shots_list = [0, 4]
    together = False

    results_dir = "./results/self_reflection_results/"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    for model, model_name, model_file in zip(models, model_names, model_files):
        for language in ["Java", "Python"]: 
            if model_name in ["deepseek-coder-33b-instruct", "CodeLlama-70b-hf"]: together = True
            for num_shots in num_shots_list:
                if model_name == "CodeLlama-70b-hf" and num_shots == 0: continue
                if reflective_logit and model_name != "gpt-3.5-turbo": continue
                for T in [0, 0.7]:
                    if T == 0: num_prompts = 500
                    else: num_prompts = 100

                    results = {} 
                    examples, queries_info = generate_prompts(num_shots, language, [model_file], num_prompts, reflective_logit)
                    if num_shots == 0:
                        examples_prompt = []
                    else:
                        examples_prompt = [[LmChatTurn(role="user", content=example['user']), 
                                            LmChatTurn(role="assistant", content=example['assistant'])] 
                                            for example in examples]
                    # Flatten
                    examples_prompt = [item for sublist in examples_prompt for item in sublist]

                    results[num_shots] = []
                    bad_out_count = 0
                    for i, query_info in tqdm(enumerate(queries_info)):
                        query = query_info['query']
                        query_prompt = [LmChatTurn(role="user", content=query)]
                        full_prompt = examples_prompt + query_prompt      
                        llm_rating, preds = prompt_lm(model, full_prompt, T, together, reflective_logit)
                        if not reflective_logit and (llm_rating is None or llm_rating not in ["1", "2", "3", "4"]):
                            bad_out_count += 1
                            print(f"Model produced bad output {bad_out_count} times:", llm_rating, preds[0]["completion"])
                            llm_rating = None
                        results[num_shots].append({
                            "LLM Rating": llm_rating, 
                            "Full Response": [pred["completion"] for pred in preds],
                            "Query": query, 
                            "Examples": examples, 
                            "BLEU-1": query_info['BLEU-1'], 
                            "BERTScore-R": query_info['BERTScore-R'],
                            "InferSent_CS": query_info['InferSent_CS'],
                            "ROUGE-1-P": query_info['ROUGE-1-P'],
                            "ROUGE-4-R": query_info['ROUGE-4-R'],
                            "ROUGE-W-R": query_info['ROUGE-W-R'],
                            "SentenceBERT_CS": query_info['SentenceBERT_CS'], 
                            "Human Gold Summary": query_info['Human Gold Summary'], 
                            "Model Generated Summary": query_info['Model Generated Summary'], 
                            "Code": query_info['code'], 
                            "Num Shots": num_shots, 
                            "Model": query_info['Model'],
                            "Rating logprobs": [pred["rating_logprobs"] for pred in preds]
                        })
                        with open(f'./results/self_reflection_results/results_{model_name}_{language}_{num_shots}_T{T}_reflective_logit_{reflective_logit}2.json', 'w') as file:
                            json.dump(results, file)
                                                    
