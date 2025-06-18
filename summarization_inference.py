import json
import numpy as np
import math
import time
from tqdm import tqdm
import asyncio
import aiohttp
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type, RetryError, before_sleep
import requests
from lmwrapper.openai_wrapper import OpenAiModelNames, get_open_ai_lm
from lmwrapper.structs import LmPrompt
# from lmwrapper.abstract_predictor import LmPredictor
# from lmwrapper.openai_wrapper import OpenAIPredictor

inferences = 0
backoff_event = asyncio.Event()
backoff_event.set()

async def before_retry(retry_state):
    global inferences
    print(f"Backing off before retry attempt {retry_state.attempt_number}. Pausing all tasks.")
    backoff_event.clear()  # Signal all tasks to pause
    time.sleep(2 ** retry_state.attempt_number)  # Backoff delay
    backoff_event.set()  # Resume task execution
    print("Resuming all tasks.")

@retry(
    stop=stop_after_attempt(5), 
    wait=wait_exponential(multiplier=1, max=10), 
    retry=retry_if_exception_type(Exception),
    before_sleep=before_retry
)
async def together_model_inference(sample, session, contexts, results, model):
    global backoff_event
    global inferences
    # Check if a backoff is in progress and wait if so
    await backoff_event.wait()
    url = "https://api.together.xyz/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": contexts[sample]}
        ],
        "max_tokens": 100,
        "stop": ["\n"],
        "temperature": 0,
        "top_p": 1,
        "n": 1,
        "logprobs": True,
        "echo": True
    }

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": "Bearer 9c64e32d4da7da57ad79b203b28b6582c4467dca72b110793289615ad83bb751"
    }

    #ca_bundle_path = certifi.where()
    #ssl_context = ssl.create_default_context(cafile=ca_bundle_path)
    start_time = time.time()
    async with session.post(url, json=payload, headers=headers, ssl=False) as response:
        response_text = await response.text()
        json_response = json.loads(response_text)
        # prompt_log_probs = json_response["prompt"]["logprobs"]["token_logprobs"]
        completion = json_response["choices"][0]["text"]
        log_probs = json_response["choices"][0]["logprobs"]["token_logprobs"]
        probs = [math.exp(log_prob) for log_prob in log_probs]
        # prompt_probs = [math.exp(log_prob) for log_prob in prompt_log_probs]
        results[sample] = {"completion": completion, "probs": probs, "prompt_probs": 1}
        inferences += 1
        print(f"Processed {inferences}: {time.time()-start_time}")

# @retry(
#     stop=stop_after_attempt(5), 
#     wait=wait_exponential(multiplier=1, max=10), 
#     retry=retry_if_exception_type(Exception)
# )
def openai_model_inference(sample, contexts, results, model):
    global backoff_event
    global inferences
    start_time = time.time()
    lm = get_open_ai_lm(
        model_name = OpenAiModelNames.gpt_3_5_turbo
    )
    pred = lm.predict(
        LmPrompt(
            text=contexts[sample],
            max_tokens=100,
            logprobs=1,
            num_completions=1,
            temperature=0,
            stop=["\n"]
        )
    )
    response=pred.completion_text.strip()
    log_probs=pred.completion_logprobs
    probs = [math.exp(log_prob) for log_prob in log_probs]
    results[sample] = {"completion": response, "probs": probs}
    inferences += 1
    print(f"Processed {inferences}: {time.time()-start_time}")

def openai_inference(model, language, prompts_path, num_samples, start_i, out_dir, prompting):
    results = {}
    contexts = {}
    with open(prompts_path, "r") as f:
        for line in f:
            context_data = json.loads(line)
            contexts[context_data["index"]] = context_data["context"]
    
    start_time = time.time()

    for i in range(start_i, num_samples):
        openai_model_inference(i, contexts, results, model)
        if i % 100 == 0:
            write_results(out_dir, language, prompting, results, model)
            time.sleep(1)
    write_results(out_dir, language, prompting, results, model)

    end_time = time.time()
    print(f"Inference completed in {end_time - start_time} seconds")


async def hf_inference(sample, session, contexts, results, model):
    global backoff_event
    global inferences
    # Check if a backoff is in progress and wait if so
    await backoff_event.wait()
    API_TOKEN = 'YOUR_HUGGING_FACE_API_TOKEN_HERE'  # Replace with your actual token

    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    API_URL = "https://api-inference.huggingface.co/models/bigcode/starcoder2-15b"

    start_time = time.time()
    payload = "Write a function to find the nth fibonacci number:"
    async with session.post(API_URL, headers=headers, json=payload) as response:
        response_text = await response.text()
        json_response = json.loads(response_text)
        completion = json_response["choices"][0]["text"]
        log_probs = json_response["choices"][0]["logprobs"]["token_logprobs"]
        probs = [math.exp(log_prob) for log_prob in log_probs]
        results[sample] = {"completion": completion, "probs": probs}
        inferences += 1
        print(f"Processed {inferences}: {time.time()-start_time}")

def write_results(out_dir, language, prompting, results, model):
    print("writing")
    fr=open(out_dir+model.split('/')[-1]+"_"+prompting+".txt","w", encoding="utf-8")
    for index, result in sorted(results.items()):
        fr.write(str(index)+"\t\t"+result["completion"].strip()+"\t\t"+str(np.mean(result["probs"]))+"\t\t"+str(np.sum(result["probs"]))+"\t\t"+str(result["probs"])+"\n")
    print("completed")
    fr.close()

async def together_inference(model, language, prompts_path, num_samples, start_i, out_dir, prompting):
    results = {}
    contexts = {}
    with open(prompts_path, "r") as f:
        for line in f:
            context_data = json.loads(line)
            contexts[context_data["index"]] = context_data["context"]
    
    start_time = time.time()

    # Restart the session every 500 requests due instability
    num_sessions = (num_samples - start_i) // 5
    for session_i in range(num_sessions):
        async with aiohttp.ClientSession() as session:
            tasks = []
            for i in range(start_i + (session_i * 500), (start_i + (session_i * 500)) + 5):
                if i % 100 == 0:
                    await asyncio.sleep(1)  # enforce 100 requests/s rate timit
                print(f"Running inference for {i}")
                tasks.append(asyncio.create_task(together_model_inference(i, session, contexts, results, model)))

            return_values = await asyncio.gather(*tasks, return_exceptions=True)
            for i, return_val in enumerate(return_values):
                if isinstance(return_val, RetryError):
                    print(f"Task {i} failed after maximum retry attempts and will be skipped.")
                elif isinstance(return_val, Exception):
                    print(f"Task {i} failed due to an unexpected error: {return_val}")
        write_results(out_dir, language, prompting, results, model)

    end_time = time.time()
    print(f"Inference completed in {end_time - start_time} seconds")

import os, asyncio
from together import AsyncTogether
async def async_chat_completion(messages, model, indices):
    results = {}
    async_client = AsyncTogether(api_key=os.environ.get("TOGETHER_API_KEY"))
    tasks = [
        async_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": message}],
            max_tokens=100,
            stop=[],
            temperature=0,
            top_p=1,
            n=1,
            logprobs=True,
            echo=True
        )
        for message in messages
    ]
    responses = await asyncio.gather(*tasks)

    for i, response in enumerate(responses):
        completion = response.choices[0].message.content
        log_probs = response.choices[0].logprobs.token_logprobs
        prompt_log_probs = response.prompt[0].logprobs.token_logprobs
        probs = [math.exp(log_prob) for log_prob in log_probs if log_prob is not None]
        prompt_probs = [math.exp(log_prob) for log_prob in prompt_log_probs if log_prob is not None]
        results[indices[i]] = {"completion": completion, "probs": probs, "prompt_probs": prompt_probs}
    return results


async def together_inference_v2(model, language, prompts_path, num_samples, start_i, out_dir, prompting):
    contexts = {}
    with open(prompts_path, "r") as f:
        for line in f:
            context_data = json.loads(line)
            contexts[context_data["index"]] = context_data["context"]
    
    start_time = time.time()
    batch_size = 5
    results = {}

    for batch_start in range(start_i, num_samples, batch_size):
        print(f"Processing batch {batch_start}")
        batch_end = min(batch_start + batch_size, num_samples)
        messages = [contexts[i] for i in range(batch_start, batch_end)]
        indices = [i for i in range(batch_start, batch_end)]
        batch_results = await async_chat_completion(messages, model, indices)
        results.update(batch_results)
        if batch_start % 300 == 0 and batch_start != 0: 
            write_results(out_dir, language, prompting, results, model)
            await asyncio.sleep(40)  # Wait for 1 minute before processing the next batch

    write_results(out_dir, language, prompting, results, model)
    end_time = time.time()
    print(f"Inference completed in {end_time - start_time} seconds")


    def hf_inference_batch(messages, results, model):
        results = {}
        responses = []
        for message in messages:
            API_URL = "https://v6qbbjso2ig9nx7q.us-east-1.aws.endpoints.huggingface.cloud"
            headers = {
                "Accept" : "application/json",
                "Authorization": "Bearer YOUR_HUGGING_FACE_API_TOKEN_HERE",  # Replace with your actual token
                "Content-Type": "application/json" 
            }

            def query(payload):
                response = requests.post(API_URL, headers=headers, json=payload)
                return response.json()

            output = query({
                "inputs": message,
                "parameters": {
                    "temperature": 0.000001,
                    "max_new_tokens": 100
                }
            })
            responses.append(output)

        for i, response in enumerate(responses):
            completion = response.choices[0].message.content
            log_probs = response.choices[0].logprobs.token_logprobs
            prompt_log_probs = response.prompt[0].logprobs.token_logprobs
            probs = [math.exp(log_prob) for log_prob in log_probs if log_prob is not None]
            prompt_probs = [math.exp(log_prob) for log_prob in prompt_log_probs if log_prob is not None]
            results[indices[i]] = {"completion": completion, "probs": probs, "prompt_probs": prompt_probs}
        return results

    contexts = {}
    with open(prompts_path, "r") as f:
        for line in f:
            context_data = json.loads(line)
            contexts[context_data["index"]] = context_data["context"]
    
    start_time = time.time()
    batch_size = 5
    results = {}

    for batch_start in range(start_i, num_samples, batch_size):
        print(f"Processing batch {batch_start}")
        batch_end = min(batch_start + batch_size, num_samples)
        messages = [contexts[i] for i in range(batch_start, batch_end)]
        indices = [i for i in range(batch_start, batch_end)]
        batch_results = hf_inference_batch(messages, model, indices)
        results.update(batch_results)
        if batch_start % 300 == 0 and batch_start != 0: 
            write_results(out_dir, language, prompting, results, model)
            await asyncio.sleep(40)  # Wait for 1 minute before processing the next batch

    write_results(out_dir, language, prompting, results, model)
    end_time = time.time()
    print(f"Inference completed in {end_time - start_time} seconds")

def hf_endpoint_inference(prompts_path, out_dir, language, prompting, model):
    results = {}
    contexts = {}
    with open(prompts_path, "r") as f:
        for line in f:
            context_data = json.loads(line)
            contexts[context_data["index"]] = context_data["context"]
    

    for i, context in contexts.items():
        if i<104:
            continue
        print(f"Running inference for {i}")
        API_URL = "https://yhe244rxva61m1gk.us-east-1.aws.endpoints.huggingface.cloud/"
        headers = {
            "Accept" : "application/json",
            "Authorization": "Bearer YOUR_HUGGING_FACE_API_TOKEN_HERE",  # Replace with your actual token
            "Content-Type": "application/json" 
        }

        def query(payload):
            response = requests.post(API_URL, headers=headers, json=payload)
            return response.json()
        
        try:
            output = query({
                    "inputs": context,
                    "parameters": {
                        "adapter_id": "null",
                        "best_of": 1,
                        "decoder_input_details": False,
                        "details": True,
                        "do_sample": True,
                        # "top_k": 10,
                        # "top_p": 0.95,
                        "max_new_tokens": 100,
                        "temperature": None,
                        # "top_n_tokens": 5,
                        # "truncate": None,
                        # "typical_p": 0.95,
                        # "watermark": True,
                        # "top_logprobs": 5,
                        "return_full_text": False,
                        "stop":["\n"]
                    },
                    "stream": False
                })
            log_probs = []
            tokens = []
            for token in output[0]['details']['tokens']:
                log_probs.append(token['logprob'])
                tokens.append(token['text'])
            results[int(i)] = {"completion": output[0]["generated_text"], "probs": log_probs}
            write_results(out_dir, language, prompting, results, model)
        except Exception as e:
            print(f"Error processing {i}: {e}")


if __name__ == "__main__":
    hf_endpoint_inference(prompts_path="./data/Python/prompting_data/haque_BM25_contexts.jsonl",
                          out_dir="./data/Python/model_outputs/",
                          language="Python",
                          prompting="haque_BM25",
                          model="CodeLlama-70b-hf")
    # asyncio.run(together_inference_v2(model="deepseek-ai/deepseek-coder-33b-instruct",
    #                      language="Python",
    #                      prompts_path="./data/Python/prompting_data/haque_BM25_contexts.jsonl",
    #                      num_samples=210,
    #                      start_i=0,
    #                      out_dir="./data/Python/model_outputs/",
    #                      prompting="haque_BM25"))   
    # Example runs:
    # asyncio.run(together(model="deepseek-ai/deepseek-coder-33b-instruct",
    #                      language="Java",
    #                      prompts_path="./data/Java/prompting_data/ASAP_contexts.jsonl",
    #                      num_samples=5000,
    #                      start_i=0,
    #                      out_dir="./data/Java/model_outputs/",
    #                      prompting="ASAP"))
    # openai_inference(model="openai/gpt-3.5-turbo2",
    #                      language="Python",
    #                      prompts_path="./data/Python/prompting_data/ASAP_contexts.jsonl",
    #                      num_samples=5000,
    #                      start_i=4901,
    #                      out_dir="./data/Python/model_outputs/",
    #                      prompting="ASAP")
