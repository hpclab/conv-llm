#%%
from methods.pyterrier_methods import *
from os.path import join as jp
import argparse
#import psutil
def transform(rewritings, retriever,qrels,eval_metrics ):
    rewritings = rewritings[['qid','query']]
    results = retriever.transform(rewritings)
    #results.to_csv(f'results/bing_rewritings/{name}.csv')
    results_per_query = pt.Experiment([results], rewritings, qrels, names=['DPH'], 
                eval_metrics = eval_metrics,perquery=True)
    #results_per_query.to_csv((f'results/bing_rewritings/{name}_per_query.csv'))
    results_mean = pt.Experiment([results], rewritings, qrels, names=['DPH'], 
                eval_metrics = eval_metrics)
    #results_mean.to_csv((f'results/bing_rewritings/{name}_mean.csv'))  
    #results.to_csv((f'results/bing_rewritings/{name}.csv'), index=False)
    return results, results_mean,results_per_query
#%%
### LOAD EVALUATION AND QRELS

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trec_path", type=str, help="collection path", default="./trec/")
    parser.add_argument("--rewriting_path",type =str, help='folder containing rewritings', default = 'data/rewritings/cleaned/')
    parser.add_argument("--year",type =int, help='year of cast', default = 2019)
    parser.add_argument("--outpath",type =str, help='folder where to save the reranked files', default = 'data/results/')
    parser.add_argument("--index_path",type =str, help='index path', default = '/data5/conversational/indexes/cast_2019_2020_stemmed_index/')
    #parser.add_argument("--prev_questions",type =bool, help='Use previous questions', default = False)
    args = parser.parse_args()
    year =args.year
    trec_path = args.trec_path if args.trec_path[-1]=='/' else args.trec_path+'/'
    evaluation_path = trec_path+'treccast/'
    print(evaluation_path)
    qrels_path = trec_path+'qrels/'
    print(qrels_path)
    all_evaluation = load_evaluation(evaluation_path,load_train=False).reset_index(drop=True)
    all_qrels = load_all_qrels(qrels_path).reset_index(drop=True)
    qrels = all_qrels[all_qrels.year == year]
    evaluation = all_evaluation[all_evaluation.qid.isin(qrels.qid.unique())]
    if year ==2019:
        manual = pd.read_csv(f'{evaluation_path}test_manual_utterance.tsv', sep = '\t', names = ['qid','query'])
        manual = manual[manual.qid.isin(qrels.qid.unique())]
        manual['query'] = manual['query'].apply(terrier_query)
    elif year == 2020:
        manual = create_df_from_json(pd.read_json(f'{evaluation_path}2020_manual_evaluation_topics_v1.0.json'))
        manual = manual.rename(columns = {'number':'qid', 'manual_rewritten_utterance':'query'})
        manual = manual[manual.qid.isin(qrels.qid.unique())]
        manual['query'] = manual['query'].apply(terrier_query)
    elif year == 2021:
        manual = create_df_from_json(pd.read_json(f'{evaluation_path}2021_manual_evaluation_topics_v1.0.json'))
        manual = manual.rename(columns = {'number':'qid', 'manual_rewritten_utterance':'query'})
        manual = manual[manual.qid.isin(qrels.qid.unique())]
        manual['query'] = manual['query'].apply(terrier_query)    
    index_path =  args.index_path
    print('iundex_path',index_path)
    index = trec_index(index_path)
    eval_metrics = ['map_cut_200','map_cut_1000','recip_rank','P_3','P_1','ndcg_cut_3','ndcg_cut_1000','recall_200','recall_1000','recall_500']
    DPH = pt.BatchRetrieve(index, wmodel='DPH',verbose=True)
    retrieval_method = [DPH]
    baseline = evaluation
    baseline['query']=evaluation['raw_utterance'].apply(terrier_query)
    output_path = args.outpath if args.outpath[-1]=='/' else args.outpath+'/'
    ###CHECK IF THERE ARE SUBFOLDERS
    if not(os.path.isdir(output_path+'for_reranking/')):
        os.mkdir(output_path+'for_reranking/')
    if not(os.path.isdir(output_path+'mean/')):
        os.mkdir(output_path+'mean/')
    if not(os.path.isdir(output_path+'per_query/')):
        os.mkdir(os.path.join(output_path,'per_query/'))        
    ###CHECK IF RESULTS FILE ARE ALREADYTHERE    
    if not(os.path.isfile(output_path+f'/for_reranking/Manual_{str(year)}.tsv')):
        print('Evaluating Manual')
        baseline_manual,baseline_manual_mean,baseline_manual_per_query = transform(manual,DPH,qrels,eval_metrics)   
        baseline_manual_mean.insert(0,'type',str(year)+'_Manual') 
        baseline_manual_mean.to_csv(output_path + f'mean/Manual_{str(year)}.csv')
        baseline_manual.to_csv(output_path + f'for_reranking/Manual_{str(year)}.tsv', header=None, sep='\t')
        baseline_manual_per_query.to_csv(os.path.join(output_path,f'per_query/Manual_{str(year)}.tsv'), header=None, sep='\t')
    else:
        #baseline_manual_mean = pd.read_csv(output_path + 'mean/'+str(year)+'_Manual.csv', index_col=0)
        baseline_manual = pd.read_csv(os.path.join(output_path,f'for_reranking/Manual_{str(year)}.tsv'), index_col=0,sep = '\t',names= ['qid','docid','docno','rank','score','query'])
        baseline_manual_mean = pt.Experiment([baseline_manual], manual, qrels, names=['DPH'], 
                eval_metrics = eval_metrics)
        baseline_manual_mean.insert(0,'type',str(year)+'_Manual') 
        baseline_manual_mean.to_csv(output_path + f'mean/Manual_{str(year)}.csv')
        #baseline_manual_per_query = pd.read_csv(os.path.join(output_path,'per_query/Manual.tsv'), index_col=0,sep = '\t',names= ['type','name','qid','measure','value'])
        

    if not(os.path.isfile(output_path + f'for_reranking/Original_{str(year)}.tsv')):
        baseline_raw, baseline_raw_mean, baseline_raw_per_query = transform(baseline,DPH,qrels,eval_metrics)   
        baseline_raw_mean.insert(0,'type',f'Original_{str(year)}')
        baseline_raw_mean.to_csv(output_path + f'mean/Original_{str(year)}.csv')
        baseline_raw.to_csv(output_path + f'for_reranking/Original_{str(year)}.tsv', header=None, sep='\t')
        baseline_raw_per_query.to_csv(output_path + f'per_query/Original_{str(year)}.tsv', header=None, sep='\t')
    else:
        #baseline_raw_mean = pd.read_csv(output_path + 'mean/Original.csv', index_col=0)
        baseline_raw = pd.read_csv(output_path + f'for_reranking/Original_{str(year)}.tsv', index_col=0,sep = '\t',names= ['qid','docid','docno','rank','score','query'])
        #baseline_raw_per_query = pd.read_csv(output_path + 'per_query/Original.tsv', index_col=0,sep = '\t',names= ['type','name','qid','measure','value'])
        baseline_raw_mean = pt.Experiment([baseline_raw], baseline, qrels, names=['DPH'], 
                eval_metrics = eval_metrics)
        baseline_raw_mean.insert(0,'type',f'Original_{str(year)}')
        baseline_raw_mean.to_csv(output_path + f'mean/Original_{str(year)}.csv')        
        
    means = pd.DataFrame()
    path =args.rewriting_path#'./rewritings/prompts/'
    prompts =[x for x in os.listdir(path) if '.tsv' in x and str(year) in x]
    output_path = args.outpath
    for prompt in prompts:
        df = pd.read_csv(path+prompt, sep = '\t', names=['qid','query'])
        df =df[df.qid.isin(qrels.qid.unique())]
        #df = df.fillna('')
        if not(any(prompt == file for file in os.listdir(os.path.join(output_path,'for_reranking/')))):

            print("Elaborating prompt ",prompt)
            #df['query'] = df['query'].apply(sub_)
            df['query'] = df['query'].apply(terrier_query)
            results, results_mean,results_per_query = transform(df,DPH,qrels,eval_metrics)   
            results_mean.insert(0,'type',prompt.replace('.tsv','')+'-Guido')  
            results_mean.to_csv(output_path + 'mean/'+prompt.replace('.tsv','.csv'))
            results_per_query.to_csv(output_path + 'per_query/'+prompt, header=None, sep='\t')
            results.to_csv(output_path + 'for_reranking/'+prompt, header=None, sep='\t')
            means = pd.concat([means,results_mean])        

            #means = pd.concat([means,results_mean])
        else:
            df['query'] = df['query'].apply(terrier_query)
            results = pd.read_csv(output_path+'for_reranking/'+prompt, sep ='\t', index_col=0,names= ['qid','docid','docno','rank','score','query'])
            results_mean = pt.Experiment([results], df, qrels, names=['DPH'], eval_metrics = eval_metrics)
            results_mean.insert(0,'type',prompt.replace('.tsv','')+'-Guido')
            results_mean.to_csv(output_path + 'mean/'+prompt.replace('.tsv','.csv'))
            means = pd.concat([means,results_mean])        
    means.to_excel(output_path + f'mean_results_{str(year)}.xlsx')  
    print("First Stage Retrieval Done") 
    '''
    for file in [x for x in os.listdir(output_path + 'mean/') if str(year) in x]:         
        results_mean = pd.read_csv(output_path+'mean/'+file, index_col=0)#,sep = '\t',names= ['qid','docid','docno','rank','score','query'])
        #results_mean= pd.read_csv(output_path+'mean/'+prompt.replace('.tsv','.csv'), index_col=0)
        #results_mean = pt.Experiment([results], df, qrels, names=['DPH'], 
        #    eval_metrics = eval_metrics)
        #results_mean.insert(0,'type',file.replace('.tsv','')+'-Guido')  
        #results_mean.to_csv(output_path + 'mean/'+file.replace('.tsv','.csv'))    
        means = pd.concat([means,results_mean])        
        #means = pd.concat([means,baseline_raw_mean,baseline_manual_mean])
    '''


    
if __name__=="__main__":
    main()
else:
    main()
# %%
