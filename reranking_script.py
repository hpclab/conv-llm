
from time import time
import tqdm
import pandas as pd
import numpy as np
import os
import pickle
from methods.methods import *
import pandas as pd
import pyterrier as pt
pt.init()
from pyterrier_t5 import MonoT5ReRanker
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("castorini/monot5-base-msmarco")
#import sys
#sys.path.append("/data3/muntean/denseQE")
from src.eval_utils import evaluate_methods, load_topics, load_qrels

def limit_tokens(x,n=128):
    x = tokenizer(x,add_special_tokens=False, max_length=n, truncation=True)
    x = tokenizer.decode(x['input_ids'])
    return x

#%%
import argparse
import random
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=str, help="collection path", default="/data5/conversational/collections/CAST2019/CASTcollection.tsv")
    parser.add_argument("--rewriting_path",type =str, help='folder containing rewritings', default = './data/rewritings/cleaned/')
    parser.add_argument("--ranked_path",type =str, help='folder containing ranked files to rerank', default = './data/results/for_reranking/')
    parser.add_argument("--outpath",type =str, help='folder where to save the reranked files', default = './data/results/')
    parser.add_argument('--year',type=int, help='year', default=2020)
    #parser.add_argument("--history",type =int, help='Number of previous turns to give as prompt. Default = 1', default = 1)
    #parser.add_argument("--prev_questions",type =bool, help='Use previous questions', default = False)
    args = parser.parse_args()
    docid_passage_dict = dict() 
    year = args.year
    collection_trec = args.collection# "/data5/conversational/collections/CAST2019/CASTcollection.tsv"
    ["qid", "docno", "label"] 
    with open(collection_trec) as f:
        for line in f: 
            if len(line.split("\t")) != 2:
                print(line)
            docid, passage = line.split("\t", 1)
            docid_passage_dict[docid] = passage.replace("\t", " ").strip()
    
    print(len(docid_passage_dict))
    passage_df = pd.DataFrame({'docno' : docid_passage_dict.keys() , 'text' : docid_passage_dict.values() })
    passage_df.head()
    del docid_passage_dict
    # Load topics
    rewriting_folder_path = args.rewriting_path#'./rewritings/prompts/'
    retrieved_files_path = args.ranked_path#'./results/for_reranking/'
    rewriting_files = [x for x in os.listdir(rewriting_folder_path) if str(year) in x]
    output_path = args.outpath    
    print ('rewritings  directory',rewriting_folder_path)
    print ('files for reranking  directory',retrieved_files_path)
    print ('output directory',output_path)

    metrics = ['map_cut_200','map_cut_1000','recip_rank','P_3','P_1','ndcg_cut_3','ndcg_cut_1000','recall_200','recall_1000','recall_500']
    if not(os.path.isdir(output_path+'reranked/')):
        os.mkdir(output_path+'reranked/')
    if not(os.path.isdir(output_path+'reranked_mean/')):
        os.mkdir(output_path+'reranked_mean/')    
    #rewriting_file_path = "./rewritingsGPT/rewritings/guido_prompt2_bing_Uoriginal_A_hist.tsv"
    monoT5 = MonoT5ReRanker()
    mean = pd.DataFrame()
    qrels = load_qrels(str(year))
    topics = load_topics(str(year))
    random.shuffle(rewriting_files)
    for file in tqdm.tqdm(rewriting_files):
        if not os.path.isfile(f"{output_path}reranked/{file}"):
            print( 'file : ' , file)
            rewriting_df = pd.read_csv(rewriting_folder_path+file, sep='\t',names=['qid','query'])#, delimiter=",", header=None)
            #rewriting_df.columns = ["qid", "query"]
            rewriting_df = rewriting_df[["qid", "query"]]
            #rewriting_df['query'] = rewriting_df['query'].apply(sub_)
            #rewriting_df['query'] = rewriting_df['query'].apply(limit_tokens)
            rewriting_df = rewriting_df[rewriting_df.qid.isin(qrels.qid.unique())]
             
            retrieved_df = pd.read_csv(retrieved_files_path+file, delimiter="\t", header=None, names=['qid','docid','docno','rank','score','query'])
            #retrieved_df = retrieved_df.drop([1, 3, 4], axis=1)
            retrieved_df = retrieved_df[["qid", "docno"]]
            for_reranking_df = pd.merge(retrieved_df, passage_df, on=["docno"])
            reranking_df = pd.merge(for_reranking_df, rewriting_df, on=["qid"])
            cols = ['qid', 'query', 'docno', 'text']
            reranking_df = reranking_df[cols]
            results = monoT5.transform(reranking_df)
            results.to_csv(f"{output_path}reranked/{file}", sep = "\t", index=False)
            method_list = []
            method_name_list = []
            method_list.append(results)
            method_name_list.append(file)
            reranking_results = evaluate_methods(method_list, method_name_list, topics, qrels,metrics=metrics)
            reranking_results.insert(0,'type', file)
            reranking_results.to_csv(f'{output_path}reranked_mean/{file}')

        else:
            #topics = load_topics(year)
            results = pd.read_csv(f"{output_path}reranked/{file}", sep = "\t")
            method_list = []
            method_name_list = []
            method_list.append(results)
            method_name_list.append(file)
            reranking_results = evaluate_methods(method_list, method_name_list, topics, qrels,metrics=metrics)
            reranking_results.insert(0,'type', file)
            reranking_results.to_csv(f'{output_path}reranked_mean/{file}')

        mean=pd.concat([mean,reranking_results])
    mean.to_excel(f'{output_path}/mean_results_reranking_{str(year)}.xlsx')
    
if __name__=="__main__":

        main()

#else:
#    main()