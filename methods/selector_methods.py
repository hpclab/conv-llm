import pandas as pd
import ktrain
from ktrain import text
import os

def balance_df(df,balance_value =1):
    new_df = pd.DataFrame()
    if balance_value ==1: 
        for qid in df.qid.unique().tolist():
            part = df[df.qid == qid]
            num = len(part[part.pos==1]) if len(part[part.pos==1])< len(part[part.pos==0]) else len(part[part.pos==0])
            pos = part[part.pos == 1].sample(num)
            neg = part[part.pos == 0].sample(num)
            new_df = pd.concat([new_df,pos,neg])
    else:
        for qid in df.qid.unique().tolist():
            part = df[df.qid == qid]
            num_pos = len(part[part.pos==1]) 
            num_neg =  len(part[part.pos==0])
            if num_neg > balance_value*num_pos:
                neg = part[part.pos == 0].sample(num_pos*balance_value)
            else:
                neg = part[part.pos == 0].sample(num_neg)
            pos = part[part.pos == 1].sample(num_pos) if num_pos < num_neg else part[part.pos == 1].sample(int(num_neg/3))
            new_df = pd.concat([new_df,pos,neg])
    return new_df

def extract_oracle(df):
    recall_200_oracle = pd.DataFrame()
    ndcg_cut_3_oracle = pd.DataFrame()
    for qid in df.qid.unique():
        part = df[df.qid == qid].iloc[:]
        if part.recall_200.max() > part.recall_200_raw.max():
            row_recall = part[part.recall_200 == part.recall_200.max()]
        else:
            row_recall = part.iloc[:1]
            row_recall['expansion'] = ['']
            row_recall['recall_200'] = row_recall['recall_200_raw'].iloc[0]            
            row_recall['ndcg_cut_3'] = row_recall['ndcg_cut_3_raw'].iloc[0]
            row_recall['query'] = row_recall['raw_utterance'].iloc[0]
        recall_200_oracle = pd.concat([recall_200_oracle,row_recall])
        if part.ndcg_cut_3.max() > part.ndcg_cut_3_raw.max():
            row_ndcg = part[part.ndcg_cut_3 == part.ndcg_cut_3.max()]
        else:
            row_ndcg = part.iloc[:1]
            row_ndcg['expansion'] = ['']
            row_ndcg['recall_200'] = row_ndcg['recall_200_raw'].iloc[0]            
            row_ndcg['ndcg_cut_3'] = row_ndcg['ndcg_cut_3_raw'].iloc[0]
            row_ndcg['query'] = row_ndcg['raw_utterance'].iloc[0]
        ndcg_cut_3_oracle = pd.concat([ndcg_cut_3_oracle,row_ndcg])
    return recall_200_oracle, ndcg_cut_3_oracle

def generate_training(df, metric='recall_200',retrieval_method ='DPH'):
    best_worst = pd.DataFrame()
    column = str((metric, retrieval_method))
    if column not in df.columns: column = metric
    for qid in df.qid.unique():
        part = df[df.qid == qid]
        if part[column].max() > part[metric+'_raw'].max():
            max = part[part[column] == part[column].max()]
            max['pos'] = [1]*len(max)
            max['neg'] = [0]*len(max)
        else:
            max = pd.DataFrame()
            #max = part.iloc[:1]
            #max = part[part[column] == part[column].max()]

        min = part[part[column] == part[column].min()]
        min['pos'] = [0]*len(min)
        min['neg'] = [1]*len(min)
        best_worst = pd.concat([best_worst,max, min])
    return best_worst

def generate_training_old(df, column='recall_200'):
    best_worst = pd.DataFrame()

    for qid in df.qid.unique():
        part = df[df.qid == qid]
        if part[column].max() > part[column+'_raw'].max():
            max = part[part[column] == part[column].max()]
            max['pos'] = [1]*len(max)
            max['neg'] = [0]*len(max)
        else:
            max = pd.DataFrame()
            #max = part.iloc[:1]
            #max = part[part[column] == part[column].max()]

        min = part[part[column] == part[column].min()]
        min['pos'] = [0]*len(min)
        min['neg'] = [1]*len(min)
        best_worst = pd.concat([best_worst,max, min])
    return best_worst

from random import sample
def gen_train_validation(df,frac =0.1):
    val_conv_ids = sample(df.conv_id.unique().tolist(),int(len(df.conv_id.unique())*frac))
    train_conv_ids = [x for x in df.conv_id.unique().tolist() if x not in val_conv_ids]
    validation = df[df.conv_id.isin(val_conv_ids)]
    train = df[df.conv_id.isin(train_conv_ids)]
    train = train.fillna('')
    train = train.sample(frac=1)
    validation = validation.fillna('')
    validation = validation.sample(frac = 1)
    return train,validation

def prepare_input_old(train,training_type):
    train = train.fillna('')
    if training_type.upper() == 'C+QE':
        hyp = train['query'].tolist()
        prem = train['context'].tolist()
    if training_type.upper() == 'CQ+E':
        hyp = train['expansion'].tolist()
        prem = (train['context']+' '+train['raw_utterance']).tolist()
    if training_type.upper() == 'Q+E':
        hyp = train[['expansion']].tolist()
        prem = train[['raw_utterance']].tolist()
    x_train = list(zip(prem, hyp))
    y_train = train['pos'].values 
    return x_train,y_train

def prepare_input(train,training_type,only_input=False):
    train = train.fillna('')
    if 'query' not in train.columns:
        train['query'] = train['raw_utterance'] + ' ' + train['expansion']
    if training_type.upper() == 'C+QE':
        hyp = train['query'].tolist()
        prem = train['context'].tolist()
    if training_type.upper() == 'CQ+E':
        hyp = train['expansion'].tolist()
        prem = (train['context']+' '+train['raw_utterance']).tolist()
    if training_type.upper() == 'Q+E':
        hyp = train[['expansion']].tolist()
        prem = train[['raw_utterance']].tolist()
    x_train = list(zip(prem, hyp))
    if not(only_input):
        y_train = train['pos'].values      
        return x_train,y_train
    else:
        return x_train

def train_classifier(df, training_type = 'C+QE', label_columns = ['neg', 'pos'], output ='' ,MODEL_NAME = 'bert-base-uncased' , batch_size =16, 
                     lr = 5e-5,epochs =8, class_weight = None , find_lr =False, fit_one_cycle=False):
    assert training_type.upper() in  ['C+QE','CQ+E','Q+E'], " You can train with one of this configurations 'C+QE','CQ+E','Q+E'"
    #output = 'models/' +training_type.upper().replace('+','_')
    train,validation = gen_train_validation(df)
    x_train,y_train =  prepare_input(train,training_type)
    x_test,y_test =  prepare_input(validation,training_type)
    t = text.Transformer(MODEL_NAME, maxlen=256, class_names=label_columns)
    trn = t.preprocess_train(x_train, y_train)
    val = t.preprocess_test(x_test, y_test)
    model = t.get_classifier()
    learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=batch_size) # lower bs if OOM occurs
    if not(os.path.isdir(output)):
        os.mkdir(output)
    if not(os.path.isdir(output+'/checkpoints')):
        os.mkdir(output+'/checkpoints')
    if find_lr:
        learner.lr_find()
        plot = learner.lr_plot(return_fig = True)
        plot.savefig(output+'/lr_plot.png')
    else:
        if fit_one_cycle:
            for i in range(epochs):
                print(f'Epoch {i+1}/{epochs}')
                learner.fit_onecycle(lr, 1 ,class_weight = class_weight)
                predictor = ktrain.get_predictor(learner.model, preproc=t)
                predictor.save(f'{output}/epoch_{i}')        
        else:
            learner.fit(lr,n_cycles = epochs,class_weight = class_weight, early_stopping = 8, checkpoint_folder = output+'/checkpoints')
        predictor = ktrain.get_predictor(learner.model, preproc=t) 
        predictor.save(output)
        predictions = predictor.predict(x_test)
        y_test_text = ['pos' if x ==1 else 'neg' for x in y_test]
        accuracy = len([x for x,y in zip(predictions,y_test_text) if x ==y])/len(predictions)
        with open(f'{output}/accuracy.txt', 'w') as f:
            f.write(str(accuracy))
        return learner
    
import random

def create_splits(df, split_num = 4, output_path = 'numberbatch'):
    output_path = output_path
    if not(os.path.isdir(output_path)):
        os.mkdir(output_path)
        os.mkdir(f'{output_path}/train/')
        os.mkdir(f'{output_path}/test/')
    if 'conv_id' not in df.columns:
        df['conv_id'] = [x.split('_')[0] for x in df.qid.tolist()]
    if 'turn' not in df.columns:
        df['turn'] = [x.split('_')[1] for x in df.qid.tolist()]
    df= df.reset_index()
    unique_conv = df.conv_id.unique().tolist()
    step_size = int(len(unique_conv)/split_num)
    random.shuffle(unique_conv)
    folds = [unique_conv[i:i+step_size]  for i in range(0,len(unique_conv),step_size)]
    folds_train_dfs = {f'fold_{i}' : pd.DataFrame() for i in range(1,split_num+1)}
    folds_test_dfs = {f'fold_{i}' : pd.DataFrame() for i in range(1,split_num+1)}
    for i,fold in enumerate(folds):
        folds_test_dfs[f'fold_{i+1}'] = df[df.conv_id.isin(fold)]
        folds_train_dfs[f'fold_{i+1}'] = df.drop(df[df.conv_id.isin(fold)].index)
    for key in folds_train_dfs.keys():
        folds_train_dfs[key].to_csv(f'{output_path}/train/{key}.csv')
        folds_test_dfs[key].to_csv(f'{output_path}/test/{key}.csv')    
    return   folds_train_dfs, folds_test_dfs
       
def get_predictions(model_path,x_test):
    predictor = ktrain.load_predictor(f'{model_path}')
    predicted_probs = predictor.predict(x_test, return_proba=True)
    predicted_probs = list([list(x) for x in predicted_probs])
    pos_probs = [x[1] for x in  predicted_probs]
    predicted_labels = [0 if x[1]<0.5 else 1 for x in  predicted_probs]
    return predicted_labels, pos_probs
       
def test_model(model_path,data_path):
    if model_path[-1] != '/' : model_path += '/'
    if data_path[-1] != '/' : data_path += '/'    
    if 'C_QE' in model_path or 'C+QE' in model_path: test_type = 'C+QE'
    elif 'CQ_E' in model_path or 'CQ+E' in model_path: test_type = 'CQ+E'
    test =  pd.DataFrame()
    for test_fold in range(len(os.listdir(data_path))):
        test = pd.concat([test,pd.read_csv(data_path+'test/fold_'+str(test_fold+1)+'.csv',index_col =0 )])
    x_test,y_test = prepare_input(test,test_type)
    metrics = pd.DataFrame()
    folds =[x for x in os.listdir(model_path) if 'fold' in x]
    #for fold in range(len(os.listdir(model_path))):
    for fold in folds:
        predictor = ktrain.load_predictor(f'{model_path}{fold}/')
        predicted_probs = predictor.predict(x_test, return_proba=True)
        predicted_probs = list([list(x) for x in predicted_probs])
        predicted_labels = [0 if x[1]<0.5 else 1 for x in  predicted_probs]
        accuracy = len([y  for x,y in zip(y_test,predicted_labels) if x == y])/len(y_test)
        true_pos = len([y  for x,y in zip(y_test,predicted_labels) if x == y and y==1])
        true_neg = len([y for x,y in zip(y_test,predicted_labels) if x == y and y==0])
        false_pos = len([y  for x,y in zip(y_test,predicted_labels) if x != y and y==1])
        false_neg = len([y  for x,y in zip(y_test,predicted_labels) if x != y and y==0])
        f1_score = (2*true_pos)/(2*true_pos+false_pos+false_neg) if (2*true_pos+false_pos+false_neg)!=0 else 'f1_score division by zero'
        pos_recall = true_pos/(true_pos+false_pos) if (true_pos+false_pos)!=0 else 'true_pos+false_pos=0?'
        neg_recall = true_neg/(true_neg+false_neg) if (true_neg+false_neg)!=0 else 'true_neg+false_neg=0?'
        test[f'predicted_labels_fold_{fold}'] = predicted_labels
        metrics = pd.concat([metrics,pd.DataFrame({'best_model_for_fold_n': [fold],'accuracy' : [accuracy],'f1_score' : [f1_score],'neg_recall' : [neg_recall],'pos_recall' : [pos_recall]})])
    return test,metrics


def get_metrics(y_test,predicted_labels):
        accuracy = len([y  for x,y in zip(y_test,predicted_labels) if x == y])/len(y_test)
        true_pos = len([y  for x,y in zip(y_test,predicted_labels) if x == y and y==1])
        true_neg = len([y for x,y in zip(y_test,predicted_labels) if x == y and y==0])
        false_pos = len([y  for x,y in zip(y_test,predicted_labels) if x != y and y==1])
        false_neg = len([y  for x,y in zip(y_test,predicted_labels) if x != y and y==0])
        f1_score = (2*true_pos)/(2*true_pos+false_pos+false_neg) if (2*true_pos+false_pos+false_neg)!=0 else 'f1_score division by zero'
        pos_recall = true_pos/(true_pos+false_pos) if (true_pos+false_pos)!=0 else 'true_pos+false_pos=0?'
        neg_recall = true_neg/(true_neg+false_neg) if (true_neg+false_neg)!=0 else 'true_neg+false_neg=0?'
        return {'accuracy' : [accuracy],'f1_score' : [f1_score],'neg_recall' : [neg_recall],'pos_recall' : [pos_recall]}
    
    
    

def test_predictions(model_path,df):
    if model_path[-1]!= '/': model_path+='/'
    if 'C_QE' in model_path: test_type = 'C+QE'
    elif 'CQ_E' in model_path: test_type = 'CQ+E'
    metrics = pd.read_csv(model_path+'metrics.csv',index_col=0)
    best_fold = metrics[metrics.f1_score == metrics.f1_score.max()]['best_model_for_fold_n'].values[0]
    model_path += 'fold_'+str(best_fold)
    x_test = prepare_input(df,test_type, only_input=True)
    predicted_labels, pos_probs = get_predictions(model_path,x_test)
    df['positive_probability'] = pos_probs
    return df


def extract_best_predictions_old(df):
    predictions = pd.DataFrame()
    for conv_id in df.conv_id.unique():
        part_conv = df[df.conv_id == conv_id].iloc[:]
        for turn in part_conv.turn.unique():
            part_turn = part_conv[part_conv.turn == turn]
            predictions = pd.concat([predictions, part_turn[part_turn.positive_probability == part_turn.positive_probability.max()].iloc[:1]])
    return predictions

def extract_best_predictions(df, column = 'positive_probability'):
    predictions = pd.DataFrame()
    for qid in df.qid.unique():
        part_turn = df[df.qid == qid]
        predictions = pd.concat([predictions, part_turn[part_turn[column] == part_turn[column].max()].iloc[:1]])
    return predictions

import nltk
from nltk.stem import PorterStemmer

def is_there_new_knowledge(df, do_stemming = True):
    ps = PorterStemmer()
    df_with_new_knowledge = pd.DataFrame()
    for qid in df.qid.unique():
        part = df[df.qid == qid].iloc[:]
        original_context = part.context.iloc[0] + ' ' + part.raw_utterance.iloc[0]
        if do_stemming:
            context = [ps.stem(word) for word in nltk.tokenize.word_tokenize(original_context)]
        else:
            context = nltk.tokenize.word_tokenize(original_context)
        presence_values = []
        for row in range(len(part)):
            tokenized_expansion = nltk.tokenize.word_tokenize(part.expansion.iloc[row])
            if do_stemming:
                tokenized_expansion = [ps.stem(x) for x in tokenized_expansion]
            isin = 0
            for el in tokenized_expansion:
                if el in context: isin+=1
            if isin == len(tokenized_expansion):
                presence_values.append(1)
            else:
                presence_values.append(0)
        part['presence_value'] = presence_values
        df_with_new_knowledge = pd.concat([df_with_new_knowledge,part])
    return df_with_new_knowledge