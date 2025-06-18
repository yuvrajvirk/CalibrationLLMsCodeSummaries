import argparse
import os
import json
from rank_bm25 import BM25Okapi
from time import sleep
import numpy as np
import re
import math
import time
import pandas as pd

def load_data():
    # Get ratings data
    ratings_df = pd.read_csv("data/haque_et_al/final_megafile.csv") 
    agg_ratings = {'function_id': [], 'similarity': [], 'adequacy': [], 'reference': [], 'generated': []}
    fids = ratings_df['function_id'].unique()
    for i in range(len(fids)):
        fid = fids[i]
        df_baseline = ratings_df[(ratings_df['function_id'] == fid) & (ratings_df['source'] == 'baseline')]
        df_reference = ratings_df[(ratings_df['function_id'] == fid) & (ratings_df['source'] == 'reference')]
        agg_ratings['function_id'].append(fid)
        agg_ratings['similarity'].append(np.mean(df_baseline['similarity']))
        agg_ratings['adequacy'].append(5 - np.mean(df_baseline['adequate']))
        agg_ratings['reference'].append(df_reference['text'].iloc[0])
        agg_ratings['generated'].append(df_baseline['text'].iloc[0])
    agg_ratings_df = pd.DataFrame.from_dict(agg_ratings)
    
    # Get functions data
    with open("data/haque_et_al/functions.json", "r") as f:
        functions_data = json.load(f)
    functions_df = pd.DataFrame.from_dict(functions_data, orient='index')
    functions_df.reset_index(inplace=True)
    functions_df.rename(columns={'index': 'function_id'}, inplace=True)

    # Merge data
    functions_df['function_id'] = functions_df['function_id'].astype(int)
    functions_df.rename(columns={0: 'function'}, inplace=True)
    merged_df = pd.merge(agg_ratings_df, functions_df, on='function_id')

    return merged_df


if __name__ == "__main__":
    # Get functions
    merged_df = load_data()
    functions = merged_df['function'].tolist()
    references = merged_df['reference'].tolist()
    # Extract repo: No repos
    # Extract DFGs

    # Extract IDs
    # Form Prompt
    # Save prompts

    model = "davinci"
    language = "Python"
    data_folder= "./data/Python/prompting_data"
    out_path_bm25 = "./data/Python/prompting_data/haque_BM25_contexts.jsonl"
    mode = "BM25"
    number_of_fewshot_sample = 3
    use_repo = "no"
    use_id = "no"
    use_dfg = "no"
    name = ""

    tokenized_corpus = [doc.split(" ") for doc in functions]
    bm25 = BM25Okapi(tokenized_corpus)

    is_error=0
    error_count=0

    for i in range(len(functions)):
        try:
            context_bm25=""  
            context_asap=""     
            query = functions[i]
            if mode=="BM25":
                if is_error==0:
                    tokenized_query = query.split(" ")
                    x=bm25.get_scores(tokenized_query)   
                    arr = np.array(x)
                    x=arr.argsort()[-int(number_of_fewshot_sample+1):][::-1][1:]
                if (error_count%4==0 and error_count>0) and len(x)>1:
                    x=x[0:len(x)-1]
                    is_error=0
                
                for w in x:
                    context_bm25=context_bm25+functions[w].strip()+"\n"
                    context_asap=context_asap+functions[w].strip()+"\n"
                    # if use_repo=="yes":
                    #     context_asap=context_asap+train_repo[w].strip()+"\n"                   
                    # if use_id=="id3":
                    #     context_asap=context_asap+train_id3[w].strip()+"\n"                                                
                    # if use_dfg=="yes":
                    #     context_asap=context_asap+train_dfg[w].strip()+"\n"                        
                        
                        
                    context_asap=context_asap+"Write down the original comment written by the developer.\n"
                    context_asap=context_asap+"Comment: "+references[w]+"\n\n"

                    context_bm25=context_bm25+"Write down the original comment written by the developer.\n"
                    context_bm25=context_bm25+"Comment: "+references[w]+"\n\n"
            
            context_bm25=context_bm25+functions[i].strip()+"\n"
            context_asap=context_asap+functions[i].strip()+"\n"
            
            # if use_repo=="yes":
            #     context_asap=context_asap+test_repo[i].strip()+"\n"
            # if use_dfg=="yes":
            #     context_asap=context_asap+test_dfg[i].strip()+"\n"                
            # if use_id=="id3":
            #     context_asap=context_asap+test_id3[i].strip()+"\n"  
                 
                
            context_bm25=context_bm25+"Write down the original comment written by the developer.\n"
            context_bm25=context_bm25+"Comment:"
            context_asap=context_asap+"Write down the original comment written by the developer.\n"
            context_asap=context_asap+"Comment:"
            with open(out_path_bm25, "a") as json_file:
                json.dump({"index": i, "context": context_bm25}, json_file)
                json_file.write("\n")
            # with open(out_path_asap, "a") as json_file:
            #     json.dump({"index": i, "context": context_asap}, json_file)
            #     json_file.write("\n")

            i=i+1
            error_count=0
            is_error=0
            print(i)
        except:
            print("error")
            is_error=1
            error_count=error_count+1
            print(error_count)
            if error_count==12:
                 fr=open(language+'_result/'+model+"_"+mode+name+".txt","a", encoding="utf-8")
                 fr.write(str(i)+"\t"+""+"\n")
                 fr.close()
                 i=i+1
                 error_count=0
                 is_error=0
            sleep(10)
            continue