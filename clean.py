

#%%
import pandas as pd
import re
import os

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("castorini/monot5-base-msmarco")

def truncate_sents_512(x):
    x = tokenizer(x,add_special_tokens=False, max_length=512, truncation=True)
    x = tokenizer.decode(x['input_ids'])
    return x


def sub_(x):
    x = re.sub('\n- .*','',x)
    x = re.sub('De-contextualized rewrite under the multi-turn information-seeking dialog context:','',x)
    x = re.sub('Response:.*','',x)
    x = re.sub('\nCurrent question.*\n.*','',x)
    x = re.sub('Previous question:.*\nRewritten.*','',x)
    x = re.sub('\t.*sorry.*','',x)    
    x = re.sub('"Earlier.*\.','',x)    
    x = re.sub('Earlier, we.*\.','',x)    
    x = re.sub('Keywords added:.*','',x)
    x = re.sub('"keywords:.*','',x)
    x = re.sub('Response:.*','',x)
    x = re.sub('User: .*','',x)
    x = re.sub('AI assistant:.*','',x)
    x = re.sub('Response: .*','',x)
    x = re.sub('"Current question:.*','',x)
    x = re.sub('Current question:.*','',x)
    x = re.sub('"Context: ','',x)
    x = re.sub('Context: ','',x)
    x = re.sub('','',x)
    x = re.sub('','',x)
    x = re.sub('"Reformulated question:','',x)
    x = re.sub('Reformulated question:','',x)
    x = re.sub('Reformulated question : ','',x)
    x = re.sub('Reformulated question: ','',x)
    x = re.sub('"Rephrased question: ','',x)
    x = re.sub('Rephrased question: ','',x)
    x = re.sub('"Request for conversational system: .*\n\nRewritten request: "','',x)
    x = re.sub('Request for conversational system:.*\n\nRewritten request: "','',x)
    x = re.sub('Previous keywords: ','',x)
    x = re.sub('"From the previous question:.*','',x)
    x = re.sub('From the previous question:.*','',x)
    x = re.sub('"Previous context:.*','',x)
    x = re.sub('Previous context:.*','',x)
    x = re.sub('"Keywords: ','',x)
    x = re.sub('Keywords: ','',x)
    x = re.sub('\n\n','',x)
    x = re.sub('"Search keywords: ','',x)
    x = re.sub('Search keywords: ','',x)
    x = re.sub('Prompt: ','',x)
    x = re.sub('Prompt for search engine: ','',x)
    x = re.sub('Search Engine Prompt: ','',x)
    x = re.sub("I'm sorry, but your current question",'',x)
    x = re.sub('lacks sufficient context .*','',x)
    x = re.sub('Query for a search engine: ','',x)
    x = re.sub('Search prompt: ','',x)
    x = re.sub('Search engine prompt: ','',x)
    x = re.sub('Search engine prompt:','',x)
    x = re.sub('Rewritten question: ','',x)
    x = re.sub('Rewritten question:','',x)
    x = re.sub('Request for a retrieval system: ','',x)
    x = re.sub('Request for a retrieval system:','',x)
    x = re.sub('"Request for clarification: ','',x)
    x = re.sub('Request for clarification:','',x)
    x = re.sub('Request: ','',x)
    x = re.sub('"Request for clarification: ','',x)
    x = re.sub('.*answer: ','',x)
    x = re.sub('Answer:.*','',x)
    x = re.sub('Question: ','',x)
    x = re.sub('"OP:.*','',x)
    x = re.sub('Request for retrieval system: ','',x)
    x = re.sub('Revised question: ','',x)
    x = re.sub('Could you please clarify your question?','',x)
    x = re.sub('Building on the previous questions:','',x)
    x = re.sub('Building on the previous questions','',x)
    x = re.sub('Building on previously asked questions ','',x)
    x = re.sub('Reformulated question in a multi-turn information-seeking dialog context: ','',x)
    x = re.sub('Rewritten: ','',x)
    x = re.sub('Reformulated: ','',x)
    x = re.sub("Revised: ",'',x)
    x = re.sub("Rewritten question with keywords: ",'',x)
    x = re.sub("\d\..*",'',x)
    x = re.sub('\(.*\)','',x)
    x = re.sub('"','',x)
    x = re.sub('"\n','',x)
    x = re.sub('\?.*','?',x)
    x = re.sub("I'm sorry, but .*",'',x)
    x = re.sub("I'm sorry.*",'',x)
    x = re.sub("Sorry.*",'',x)
    x = re.sub('^"\n','',x)
    x = re.sub('\d\).*','',x)
    
    

    return x


def limit_tokens(x,n=128):
    x = tokenizer(x,add_special_tokens=False, max_length=n, truncation=True)
    x = tokenizer.decode(x['input_ids'])
    return x
'''
prompts = [x for x in os.listdir('./rewritings/rewritten/')]
for prompt in prompts:
    df = pd.read_csv('./rewritings/rewritten/'+prompt, sep = '\t', index_col=0)
    df = df[['qid','query']]
    df['query']=df['query'].apply(sub_)
        
    df.to_csv('./rewritings/cleaned/'+prompt,sep = '\t')
'''
# %%
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, help="collection path", default="/data4/guidorocchietti/GPT_clean/ultima_prova/rewritings/rewritten/")
    parser.add_argument("--output_folder",type =str, help='folder containing rewritings', default = '/data4/guidorocchietti/GPT_clean/ultima_prova/rewritings/cleaned/')
    parser.add_argument("--year",type =int, help='year', default = 2019)
    parser.add_argument('--sub_first_only',type =bool, action=argparse.BooleanOptionalAction,help= 'Set this option if you just want to substitute each first utterance of every conversation with the opriginal one', default=False)
    args = parser.parse_args()
    input_path = args.input_folder
    output_path = args.output_folder
    year = args.year
    sub_first_only =args.sub_first_only
    print(sub_first_only)
    if not(os.path.isdir(output_path)):
        os.mkdir(output_path)
    prompts = [x for x in os.listdir(input_path) if str(year) in x]
    if year == 2019 :
        evaluation = pd.read_csv('./trec/evaluation2019.csv')
        evaluation['turn'] = evaluation['turn'].apply(str)
        evaluation['conv_id'] = evaluation['conv_id'].apply(str)
        
    elif year == 2020 :
        evaluation = pd.read_csv('./trec/evaluation2020.csv')
        evaluation['turn'] = evaluation['turn'].apply(str)
        evaluation['conv_id'] = evaluation['conv_id'].apply(str)
    
    for prompt in prompts:
        df = pd.read_csv(input_path+prompt, sep = '\t')
        #df = df[['qid','query']]
        df['conv_id'] = [x.split('_')[0] for x in df.qid]
        df['turn'] = [x.split('_')[1] for x in df.qid]    
        if sub_first_only:      
            for conv in df['conv_id'].unique():
                df.loc[(df['conv_id']==conv) & (df['turn']=='1'), 'query'] = evaluation.loc[(evaluation['conv_id']==conv) & (evaluation['turn']=='1'), 'raw_utterance']
            df[['qid','query']].to_csv(output_path+'first_sub_'+prompt,sep='\t', index=False)
   
        else:
            df['query']=df['query'].apply(sub_)
            df = df.mask(df == '')
            df['conv_id'] = [x.split('_')[0] for x in df.qid]
            df['turn'] = [x.split('_')[1] for x in df.qid]    
            
            df = df.merge(evaluation[['qid','raw_utterance']], on='qid')
            df['query']=df['query'].fillna(df['raw_utterance'])
            #df['query'] =df['query'].apply(sub_)
            #df['query']=df['query'].fillna(df['raw_utterance'])

            df[['qid','query']].to_csv(output_path+prompt,sep='\t', index=False)
            for conv in df['conv_id'].unique():
                df.loc[(df['conv_id']==conv) & (df['turn']=='1'), 'query'] = evaluation.loc[(evaluation['conv_id']==conv) & (evaluation['turn']=='1'), 'raw_utterance']
            df[['qid','query']].to_csv(output_path+'first_sub_'+prompt,sep='\t', index=False)

            
        #df.to_csv(output_path+prompt,sep = '\t')
        
if __name__ == '__main__':
    main()
#else:main()
 
#%%
import pandas as pd 
from methods.methods import *
import sys
sys.path.append("/data3/muntean/denseQE")
from src.eval_utils import evaluate_methods, load_topics, load_qrels

#%%
import pandas as pd
import json
def extract_topics_cqr(df):
    df_topics = pd.DataFrame()
    df_topics['qid'] = df['topic_number'] + '_' + df['query_number']
    #df_topics['query'] = [terrier_query(x) for x in df.output.tolist()]
    df_topics['query'] = [(x) for x in df.output.tolist()]

    return df_topics
def read_json(file_path):
    data = []
    with open(file_path) as f:
        for line in f:
            data.append(json.loads(line))
    return data
