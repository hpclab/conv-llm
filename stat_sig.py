
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


def conv_to_paper_table(file):
    if 'prompt4' in file:
        return 'P1'
    elif 'prompt5' in file:
        return 'P2'
    elif 'prompt10' in file:
        return 'P3'
    elif 'prompt12' in file:
        return 'P4'   
    elif 'prompt14' in file:
        return 'P5' 
    elif 'promptExample' in file:
        return 'P6' 
    elif 'Example_in_history' in file:
        return 'P7'  
    elif 'promptRAR' in file:
        return 'E'
    else: return file
#%%
import argparse
metrics = ['map_cut_200','map_cut_1000','recip_rank','P_3','P_1','ndcg_cut_3','ndcg_cut_1000','recall_200','recall_1000','recall_500']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=str, help="collection path", default="/data5/conversational/collections/CAST2019/CASTcollection.tsv")
    parser.add_argument("--rewriting_path",type =str, help='folder containing rewritings', default = './data/rewritings/cleaned/')
    parser.add_argument("--ranked_path",type =str, help='folder containing ranked files to rerank', default = './data/results/reranked/')
    parser.add_argument("--outpath",type =str, help='folder where to save the reranked files', default = './data/results/')
    parser.add_argument('--year',type=int, help='year', default=2019)
    parser.add_argument('--baseline',type=str, help='chosen baseline', default='cqr')
    # python stat_sig.py --rewriting_path  ./data/rewritings/cleaned/ --ranked_path ./data/results/for_rearanking/ --year 2019 --baseline QURETEC
    #parser.add_argument("--history",type =int, help='Number of previous turns to give as prompt. Default = 1', default = 1)
    #parser.add_argument("--prev_questions",type =bool, help='Use previous questions', default = False)
    args = parser.parse_args()
    year = args.year
    chosen_baseline = args.baseline
    # Load topics
    rewriting_folder_path = args.rewriting_path#'./rewritings/prompts/'
    retrieved_files_path = args.ranked_path#'./results/for_reranking/'
    rewriting_files = [x for x in os.listdir(rewriting_folder_path) if str(year) in x]
    output_path = args.outpath    
    print ('rewritings  directory',rewriting_folder_path)
    #print ('files for reranking  directory',retrieved_files_path)
    #print ('output directory',output_path)
    qrels = load_qrels(str(year))
    topics = load_topics(str(year))
    method_list = []
    method_name_list = []

    for file in tqdm.tqdm(rewriting_files):
        if ('first_sub' in file and 'new_conv' not in file) or ('cqr' in file or 'QURETEC' in file  or 'Manual' in file or 'Original' in file):
            rewriting_df = pd.read_csv(rewriting_folder_path+file, sep='\t',names=['qid','query'])#, delimiter=",", header=None)
            rewriting_df = rewriting_df[["qid", "query"]]
            rewriting_df = rewriting_df[rewriting_df.qid.isin(qrels.qid.unique())]
            if 'ranked' not in retrieved_files_path:
                results = pd.read_csv(f"{retrieved_files_path}{file}", sep = "\t",names=['qid','docid','docno','rank','score','query'])
            else:   
                results = pd.read_csv(f"{retrieved_files_path}{file}", sep = "\t")
                
            if not chosen_baseline in file:
                method_list.append(results)
                method_name_list.append(conv_to_paper_table(file))
            else:
                method_list.insert(0,results)
                method_name_list.insert(0,conv_to_paper_table(file))
    reranking_results = pt.Experiment(method_list, 
                         topics, qrels, 
                         names=method_name_list, 
                         eval_metrics=metrics,
                         baseline=0,
                         perquery=False,
                         correction='bonferroni') 
    #reranking_results.insert(0,'type', file)
    if 'reranked' in retrieved_files_path:
        reranking_results.to_csv(f'stat_sig/stat_diff_reranked_wrt_{chosen_baseline}_{str(year)}.csv')
        reranking_results.to_excel(f'stat_sig/stat_diff_reranked_wrt_{chosen_baseline}_{str(year)}.xlsx')
    else:
        reranking_results.to_csv(f'stat_sig/stat_diff_wrt_{chosen_baseline}_{str(year)}.csv')
        reranking_results.to_excel(f'stat_sig/stat_diff_wrt_{chosen_baseline}_{str(year)}.xlsx')

    #mean=pd.concat([mean,reranking_results])
    #mean.to_excel(f'{output_path}/mean_results_reranking_{str(year)}.xlsx')
    
if __name__=="__main__":
    main()
else:  main()
# %%
