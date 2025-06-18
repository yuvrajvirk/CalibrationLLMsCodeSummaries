import argparse
import os
import json
from rank_bm25 import BM25Okapi
from time import sleep
import numpy as np
import re
import math
import time

# Based on Automatic Semantic Augmentation of Language Model Prompts code.
# https://zenodo.org/records/7793516

contexts = {}
def makestr(lst):
    p=""
    for w in lst:
        p=p+w+" "
    return p.strip()    
	#return lst
	
def main():
    # Example run
    model = "davinci"
    language = "Python"
    data_folder= "./data/Python/prompting_data"
    out_path_bm25 = "./data/Python/prompting_data/BM25_contexts.txt"
    out_path_asap = "./data/Python/prompting_data/ASAP_contexts.txt"
    mode = "BM25"
    number_of_fewshot_sample = 3
    use_repo = "yes"
    use_id = "id3"
    use_dfg = "yes"
    name = ""

    if use_repo=="yes":
        name=name+"_repo"
        
    if use_dfg=="yes":
        name=name+"_dfg"
            
    if use_id=="id3":
        name=name+"_id3"     
     
    #making necessary forlders
    if not os.path.exists(language+'_result'):
       os.makedirs(language+'_result')
    
    target_model = model

    #Reading data
    train_json = []
    for line in open(data_folder+'/train.jsonl', 'r', encoding="utf-8"):
        train_json.append(json.loads(line))
    print(len(train_json))
    
    
    test_json = []
    for line in open(data_folder+'/test.jsonl', 'r', encoding="utf-8"):
        test_json.append(json.loads(line))
    print(len(test_json)) 
    
    
    train_code=[]
    train_nl=[]
    for i in range(len(train_json)):
        train_code.append(train_json[i]['code'])
        train_nl.append(makestr(train_json[i]['docstring_tokens']))    
    
        
    test_code=[]
    test_nl=[]
    for i in range(len(test_json)):
        test_code.append(test_json[i]['code'])
        test_nl.append(makestr(test_json[i]['docstring_tokens']))   
        
    if language=="Python":
        for i in range(len(train_code)):
            m=train_code[i]
            m = re.sub(r'\"\"\"(.+?)\"\"\"', '', m)
            m = re.sub(r"\'\'\'(.+?)\'\'\'", '', m)
            m = m.replace(train_json[i]["docstring"],"")
            train_code[i]=m
        for i in range(len(test_code)):
            m=test_code[i]
            m = re.sub(r'\"\"\"(.+?)\"\"\"', '', m)
            m = re.sub(r"\'\'\'(.+?)\'\'\'", '', m)
            m = m.replace(test_json[i]["docstring"],"")
            test_code[i]=m
        
    train_repo=[]    
    with open(data_folder+"/train_repo.txt","r") as f:     
        text=f.read().strip()
        text=text.split("##########")   
        for ln in text:
            train_repo.append(ln)
            
    test_repo=[]
    with open(data_folder+"/test_repo.txt","r") as f:     
        text=f.read().strip()
        text=text.split("##########") 
        for ln in text:
            test_repo.append(ln)        
        

    train_dfg=[]    
    with open(data_folder+"/train_dfg.txt","r",encoding="utf-8") as f:     
        text=f.read().strip()
        text=text.split("##########")   
        for ln in text:
            train_dfg.append(ln)
            
    test_dfg=[]
    with open(data_folder+"/test_dfg.txt","r") as f:     
        text=f.read().strip()
        text=text.split("##########") 
        for ln in text:
            test_dfg.append(ln)        
        



    train_id3=[]    
    with open(data_folder+"/train_id_type3.txt","r") as f:     
        text=f.read().strip()
        text=text.split("##########")   
        for ln in text:
            train_id3.append(ln)
            
    test_id3=[]
    with open(data_folder+"/test_id_type3.txt","r") as f:     
        text=f.read().strip()
        text=text.split("##########") 
        for ln in text:
            test_id3.append(ln) 


    
    if mode=="BM25":
        tokenized_corpus = [doc.split(" ") for doc in train_code]
        bm25 = BM25Okapi(tokenized_corpus)
    elif mode=="fixed":
        with open(data_folder+"/fixed_3.txt","r",encoding="utf-8") as f:
            context_fixed=f.read()
    i=2315
    is_error=0
    error_count=0

    while i < (2316):
        try:
            context_bm25=""  
            context_asap=""     
            query = test_code[i]
            if mode=="BM25":
                if is_error==0:
                    tokenized_query = query.split(" ")
                    x=bm25.get_scores(tokenized_query)   
                    arr = np.array(x)
                    x=arr.argsort()[-int(number_of_fewshot_sample):][::-1]
                                
                if (error_count%4==0 and error_count>0) and len(x)>1:
                    x=x[0:len(x)-1]
                    is_error=0
                
                for w in x:
                    context_bm25=context_bm25+train_code[w].strip()+"\n"
                    context_asap=context_asap+train_code[w].strip()+"\n"
                    if use_repo=="yes":
                        context_asap=context_asap+train_repo[w].strip()+"\n"                   
                    if use_id=="id3":
                        context_asap=context_asap+train_id3[w].strip()+"\n"                                                
                    if use_dfg=="yes":
                        context_asap=context_asap+train_dfg[w].strip()+"\n"                        
                        
                        
                    context_asap=context_asap+"Write down the original comment written by the developer.\n"
                    context_asap=context_asap+"Comment: "+train_nl[w]+"\n\n"

                    context_bm25=context_bm25+"Write down the original comment written by the developer.\n"
                    context_bm25=context_bm25+"Comment: "+train_nl[w]+"\n\n"
            elif mode=="fixed":
                context=context+context_fixed
            
            
            context_bm25=context_bm25+test_code[i].strip()+"\n"
            context_asap=context_asap+test_code[i].strip()+"\n"
            
            if use_repo=="yes":
                context_asap=context_asap+test_repo[i].strip()+"\n"
            if use_dfg=="yes":
                context_asap=context_asap+test_dfg[i].strip()+"\n"                
            if use_id=="id3":
                context_asap=context_asap+test_id3[i].strip()+"\n"  
                 
                
            context_bm25=context_bm25+"Write down the original comment written by the developer.\n"
            context_bm25=context_bm25+"Comment:"
            context_asap=context_asap+"Write down the original comment written by the developer.\n"
            context_asap=context_asap+"Comment:"
            with open(out_path_bm25, "a") as json_file:
                json.dump({"index": i, "context": context_bm25}, json_file)
                json_file.write("\n")
            with open(out_path_asap, "a") as json_file:
                json.dump({"index": i, "context": context_asap}, json_file)
                json_file.write("\n")

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
    
    
    
main()
