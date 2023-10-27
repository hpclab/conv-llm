import pyterrier as pt
if not pt.started():
       pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])
from jnius import autoclass
tokeniser = autoclass("org.terrier.indexing.tokenisation.Tokeniser").getTokeniser()
import os
import pandas as pd
from tqdm import tqdm
from methods.generator_methods import *



### Takes in input the evaluation with Conversation ID, Turn id and expansions 
### and returns a DF with a single line per expansion already preprocessed and ready for retrieval phase



def prepare_for_retrieval(df):
    new_df = pd.DataFrame()
    for conv_id in tqdm(df.conv_id.unique().tolist()):
        cont = ''
        part = df[df.conv_id == conv_id]
        expansions = []
        for turn in part.turn.tolist():
            single_row = part[part.turn == str(turn)][[col for col in part.columns if 'expansions' not in col]]
            single_row['raw_utterance'] = [terrier_query(x) for x in single_row['raw_utterance'].tolist()]
            if part[part.turn == str(turn)].expansions.values[0]:
                single_row_expansions = part[part.turn == str(turn)].expansions.values[0]
            else:
                single_row_expansions = ['']
            expansions += single_row_expansions
            expansions = list(set(expansions))
            for exp in expansions:
                row = single_row.iloc[:]
                row['context'] = cont
                row['expansion'] = [terrier_query(exp.replace('_',' '))]
                new_df = pd.concat([new_df,row])
            if int(turn) <= 1 :cont += single_row.raw_utterance.values[0]
            elif int(turn) >1 : cont += ' ' + single_row.raw_utterance.values[0]
    return new_df


def prepare_for_retrieval_old(df):
    new_df = pd.DataFrame()
    for conv_id in tqdm(df.conv_id.unique().tolist()):
        cont = ''
        part = df[df.conv_id == conv_id]
        expansions = []
        for turn in part.turn.tolist():
            single_row = part[part.turn == str(turn)][[col for col in part.columns if 'expansions' not in col]]
            single_row['raw_utterance'] = [terrier_query(x) for x in single_row['raw_utterance'].tolist()]
            try:
                single_row_expansions = part[part.turn == str(turn)].expansions.values[0]
            except:
                single_row_expansions = ''
            #if not(single_row.raw_utterance.empty):
            if int(turn) == 2 :cont += single_row.raw_utterance.values[0]
            elif int(turn) >2 : cont += ' ' + single_row.raw_utterance.values[0]
            #else:
            #    print(f'en error has occurred in conversation {str(conv_id)} turn {str(turn)}')
            expansions += single_row_expansions
            expansions = list(set(expansions))
            for exp in expansions:
                row = single_row.iloc[:]
                row['context'] = cont
                row['expansion'] = [terrier_query(exp.replace('_',' '))]
                new_df = pd.concat([new_df,row])
    return new_df



### Load a PyTerrier Index if there is one otherwise it creates it
def trec_index(index_path,files=None, overwrite =False, stemmer = 'none'):
    if files == None and not overwrite:
        try:
            index_ref = pt.IndexRef.of(os.path.join(index_path,"data.properties"))
        except Exception as e:
            print(e)

    elif overwrite or not os.path.exists(os.path.join(index_path,"data.properties")):
        indexer = pt.TRECCollectionIndexer(index_path,
                                           meta= {'docno' : 49, 'text' : 4096},
                                           meta_tags = {'text' : 'ELSE'}, 
                                           verbose=True, overwrite=True,blocks=False)
        index_ref = indexer.index(files,stemmer= stemmer)
    index = pt.IndexFactory.of(index_ref)
    return index



### Takes a DF with a 'qid' and a 'query' column, a list with the retriever (already loaded) and returns
### the results for each query
def experiment(df,retrieval_method,qrels,eval_metrics, names):
    results = pt.Experiment(retrieval_method, df, qrels, names=names, 
                    eval_metrics = eval_metrics,perquery=True)
    results = results.pivot(index= 'qid', columns=['measure'], values='value').reset_index()
    results = results.merge( df,on= 'qid')
    all_results= pd.concat([all_results,results])
    return results


def experiment_per_query(df,retrieval_method :list,qrels,eval_metrics = ['recall_200','ndcg_cut_3'],names = ['DPH'] ):
    exceptions = pd.DataFrame()
    all_results = pd.DataFrame()
    for i in tqdm(range(0,len(df))):
        try:
            results = pt.Experiment(retrieval_method, df.iloc[[i]], qrels, names=names, 
                            eval_metrics = eval_metrics,perquery=True)
            results = results.pivot(index= 'qid', columns=['measure'], values='value').reset_index()
            results = results.merge( df.iloc[[i]],on= 'qid')
            all_results= pd.concat([all_results,results])
        except Exception as e:
            print(str(e))
            exceptions = pd.concat([exceptions,df.iloc[[i]]])
            results = pd.DataFrame({key : [0.0] for key in eval_metrics})
            results.insert(0,'qid',df.iloc[[i]].qid.values[0])
            results.insert(len(results.columns),'query',df.iloc[[i]]['query'].values[0])
            #results = pd.DataFrame({'qid':[df.iloc[[i]].qid],'ndcg_cut_3': [0.0],'recall_200':[0.0], 'query' : df.iloc[[i]]['query']})
            all_results= pd.concat([all_results,results])

    return all_results, exceptions

### Takes a df with at least 'qid','raw_utterance' and 'expansion' columns, the qrels and two lists
### one with the retrieval methods names and the other with the metrics names and returns the results
### per query and a df with the exceptions if any
    
def retrieve_ranked_results(df,qrels,retrieval_methods = ['DPH'],text_column = 'raw_utterance',eval_metrics = ['recall_200','ndcg_cut_3'],index_path = '../indexes/cast_2019_2020_stemmed_index', expand =False):
    
    df = df.reset_index().fillna('')
    topics = df[['qid']]
    if expand:
        print('Performing expansion, if your queries are already expanded you should set the variable "expand" equal to False')
        topics['query'] = df[text_column] + ' ' + df.expansion
    else:
        topics['query'] = df[text_column]
    index = trec_index(index_path)
    retrievers = [pt.BatchRetrieve(index, wmodel=method) for method in retrieval_methods]
    results, exceptions = experiment_per_query(topics,retrievers,qrels,eval_metrics )
    return results,exceptions

def retrieve_ranked_results_old(df,qrels,retrieval_methods = ['DPH'],eval_metrics = ['recall_200','ndcg_cut_3']):
    index_path_19_20 = '../indexes/cast_2019_2020_stemmed_index'
    index_path_21 = '../indexes/cast_2021'
    qrels_19_20 = qrels[qrels.year.isin([2019,2020])]
    qrels_21 = qrels[qrels.year.isin([2021])]
    df = df.reset_index()
    df_21 = df[df.year == 2021]
    df_19_20 = df.drop(df_21.index)
    topics_21 = df_21[['qid']]
    topics_21['query'] = df_21.raw_utterance + ' ' + df_21.expansion
    topics_19_20 = df_19_20[['qid']]
    topics_19_20['query'] = df_19_20.raw_utterance + ' ' + df_19_20.expansion
    index_19_20 = trec_index(index_path_19_20)
    index_21 = trec_index(index_path_21)
    retrievers_19_20 = [pt.BatchRetrieve(index_19_20, wmodel=method) for method in retrieval_methods]
    retrievers_21 = [pt.BatchRetrieve(index_21, wmodel=method) for method in retrieval_methods]
    results_19_20, exceptions_19_20 = experiment_per_query(topics_19_20,retrievers_19_20,qrels_19_20)
    results_21, exceptions_21 = experiment_per_query(topics_21,retrievers_21,qrels_21)
    return results_19_20,exceptions_19_20, results_21, exceptions_21 


def pt_experiment(retrievers: list, df, qrels, names, eval_metrics: list, perquery=True, verbose=True):
    results = pt.Experiment(retrievers, df, qrels, names=names,
                            eval_metrics=eval_metrics, perquery=perquery, verbose=verbose)
    results = results.pivot(
        index=['qid', 'name'], columns='measure', values='value').reset_index()
    return results

