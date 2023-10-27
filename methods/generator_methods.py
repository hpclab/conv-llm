import os
import spacy
import re
import sys
current = os.path.abspath('')
parent = os.path.dirname(current)
sys.path.append(current)
sys.path.append(parent)
import torch
import pandas as pd
from tqdm import tqdm
from torch.nn.functional import cosine_similarity
from methods.methods import *
from jnius import autoclass
tokeniser = autoclass("org.terrier.indexing.tokenisation.Tokeniser").getTokeniser()

### Takes a string and returns the tokenised version of the same string

def terrier_tokenizer(raw_utterance):
    new_utterance = " ".join(tokeniser.getTokens(raw_utterance))
    return new_utterance


### Takes a string, calls the method that tokenizes and removes stopwords and punctuation

def terrier_query(query, lowercase=True, stopwords=True, punctuation=False):
    new_query = terrier_tokenizer(query)  
    if lowercase:
        words = [word.lower() for word in new_query.split()]
        new_query = " ".join(words)
    if not punctuation:
        new_query = new_query.replace("?", " ").replace(".", "").replace("'", " ").replace("/", " ")     
    if not stopwords:
        words = [word for word in new_query.split() if word.lower() not in stopwords]
        new_query = " ".join(words)
    return new_query


### Extract Noun Chunks from a string ###
def extract_nounchunks_from_str(utterance):
    nlp = spacy.load("en_core_web_sm")
    nuncha = []
    for i in nlp(utterance).noun_chunks:
        nunch = ''
        for x in nlp(i.text):
            if x.pos_ != 'DET' and x.pos_ !='PRON':
                nunch += " " +x.text.replace(' - ','-') 
                nunch = re.sub(r'^\s+', '', nunch)
            if x.pos_ == 'NOUN' or x.pos_ == 'PROPN':
                tok = re.sub(r'^\s+', '',x.text.replace(' ','').replace(' - ','-'))
                if tok not in nuncha:
                    nuncha.append(tok)
        if nunch != '' and (nunch not in nuncha):
            nuncha.append(nunch)
    return nuncha
### Extract Noun Chunks from a list of strings ###

def extract_nounchunks(list):
    nlp = spacy.load("en_core_web_sm")
    nounchunks = []
    for utterance in list:
        nuncha = []
        for i in nlp(utterance).noun_chunks:
            nunch = ''
            for x in nlp(i.text):
                if x.pos_ != 'DET' and x.pos_ !='PRON':
                    nunch += " " +x.text.replace(' - ','-') 
                    nunch = re.sub(r'^\s+', '', nunch)
                if x.pos_ == 'NOUN' or x.pos_ == 'PROPN':
                    tok = re.sub(r'^\s+', '',x.text.replace(' ','').replace(' - ','-'))
                    if tok not in nuncha:
                        nuncha.append(tok)
            if nunch != '' and (nunch not in nuncha):
                nuncha.append(nunch)
        nounchunks.append(nuncha)
    return nounchunks

def numberbatch_top_k_similar_entities(embedding,k=5,dictionary_numberbatch=None,embedding_matrix=None):
    if embedding_matrix == None:
        embedding_matrix = torch.load('numberbatch/numberbatch_embedding_matrix.pt')
    if dictionary_numberbatch== None:
        dictionary_numberbatch = load_dict('numberbatch/numberbatch_dictionary.pkl')
    inverted_dict = {i:x for i,x in enumerate(list(dictionary_numberbatch.keys()))}
    words = []
    dist = cosine_similarity(embedding_matrix,embedding.unsqueeze(dim=0))
    for el in torch.topk(dist,k).indices:
        words.append(inverted_dict[int(el)])
    return words

def top_k_similar_from_nounchunks(nunchaku,k=5, tuple_output =False):
    import torch
    embedding_matrix = torch.load('./numberbatch/numberbatch_embedding_matrix.pt')
    dictionary_numberbatch = load_dict('./numberbatch/numberbatch_dictionary.pkl')
    nunchaku_similar_entities = []
    for line in tqdm(nunchaku):
        emb = []
        for nuncha in line:
            nuncha = nuncha.replace(' ','_')
            if nuncha in dictionary_numberbatch.keys():
                id =dictionary_numberbatch[nuncha]
                top_k = numberbatch_top_k_similar_entities(embedding_matrix[id],k=k,dictionary_numberbatch = dictionary_numberbatch,embedding_matrix=embedding_matrix)
                if tuple_output:
                    emb.append((nuncha,top_k))
                else:
                    emb += (top_k)
        nunchaku_similar_entities.append(emb)
    return nunchaku_similar_entities

def top_k_similar_from_nounchunks(nunchaku,k=5, tuple_output =False):
    import torch
    embedding_matrix = torch.load('./numberbatch/numberbatch_embedding_matrix.pt')
    dictionary_numberbatch = load_dict('./numberbatch/numberbatch_dictionary.pkl')
    nunchaku_similar_entities = []
    for line in tqdm(nunchaku):
        emb = []
        for nuncha in line:
            nuncha = nuncha.replace(' ','_')
            if nuncha in dictionary_numberbatch.keys():
                id =dictionary_numberbatch[nuncha]
                top_k = numberbatch_top_k_similar_entities(embedding_matrix[id],k=k,dictionary_numberbatch = dictionary_numberbatch,embedding_matrix=embedding_matrix)
                if tuple_output:
                    emb.append((nuncha,top_k))
                else:
                    emb += (top_k)
        nunchaku_similar_entities.append(emb)
    return nunchaku_similar_entities


### Class to manage luke model, it is possible ti instantiate different versions of luke
### and to perform task such as getting the embeddings or generate the top k similar entities
import gc
import numpy as np
class LukeEmbeddings():
    def __init__(self, task='base'):
        from transformers import LukeTokenizer

        assert task in ['base','base_large','pair_classification', 'entity_classification', 'span_classification','masked', 'masked_large']
        self.task = task
        if task == 'base' or task =='base_large':
            if task == 'base_large':task= 'large'
            from transformers import LukeModel
            self.model = LukeModel.from_pretrained(f"studio-ousia/luke-{task}")
            self.tokenizer = LukeTokenizer.from_pretrained(f"studio-ousia/luke-{task}")
        elif task == 'pair_classification':
            from transformers import LukeForEntityPairClassification
            self.model = LukeForEntityPairClassification.from_pretrained("studio-ousia/luke-large-finetuned-tacred")
            self.tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-large-finetuned-tacred")
        elif task == 'entity_classification':
            from transformers import LukeForEntityClassification            
            self.model = LukeForEntityClassification.from_pretrained("studio-ousia/luke-large-finetuned-open-entity")
            self.tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-large-finetuned-open-entity")
        elif task == 'span_classification':
            from transformers import LukeForEntitySpanClassification
            self.model = LukeForEntitySpanClassification.from_pretrained("studio-ousia/luke-large-finetuned-conll-2003")
            self.tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-large-finetuned-conll-2003")
        elif task == 'masked' or task =='masked_large':
            if task == 'masked_large': model_kind = 'large'
            else: model_kind = 'base'
            from transformers import LukeForMaskedLM
            self.model = LukeForMaskedLM.from_pretrained(f"studio-ousia/luke-{model_kind}")
            self.tokenizer = LukeTokenizer.from_pretrained(f"studio-ousia/luke-{model_kind}")
        self.entity_embeddings = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    def get_embeddings(self,text, entity_spans):
        import torch
        inputs = self.tokenizer(text, entity_spans=entity_spans, add_prefix_space=True,padding=True, return_tensors="pt")
        for el in inputs['input_ids']: #text = "Beyoncé lives in Los Angeles."
            print(self.tokenizer.convert_ids_to_tokens(el))
        inputs.to(self.device)
        self.model.to(self.device)
        outputs = self.model(**inputs,output_hidden_states=True)#,return_dict =False)
        #self.model.to('cpu')
        del inputs
        torch.cuda.empty_cache()
        return outputs
    
    def get_entity_embeddings(self, text, entity_spans, batch = 1024):
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            for i in tqdm(range(0,len(text), batch)):
                inputs = self.tokenizer(text[i:i+batch], entity_spans=entity_spans[i:i+batch], add_prefix_space=True,padding=True, return_tensors="pt")
                inputs.to(self.device)
                outputs = self.model(**inputs,output_hidden_states=True)#,return_dict =False)
                del inputs
                torch.cuda.empty_cache()
                if self.entity_embeddings == None:
                    self.entity_embeddings = outputs.entity_hidden_states[-1].to('cpu')
                else: 
                    self.entity_embeddings = torch.cat((self.entity_embeddings, outputs.entity_hidden_states[-1].to('cpu')), dim=0)
                del outputs

    def generate_related_entities(self, sentences, batch=32, number_of_words=50):
        luke = self.model
        device = self.device
        number_of_words= number_of_words
        entity_spans = [[(len(x)+1,len(x)+len(self.tokenizer.mask_token)+1)] for x in sentences]
        utterances_mask_token = [x +' '+self.tokenizer.mask_token for x in sentences]
        self.model.to(device)
        predicted_entities = []
        inverted_vocab = {i:k for i,k in enumerate(self.tokenizer.entity_vocab)}
        def id_to_tok(intero):
            return inverted_vocab[int(intero)]
        for i in tqdm(range(0,len(utterances_mask_token),batch)):
            inputs = self.tokenizer(utterances_mask_token[i:i+batch],entity_spans = entity_spans[i:i+batch], add_prefix_space=True,padding=True, return_tensors="pt")
            inputs.to(device)
            with torch.no_grad():
                outputs = self.model(**inputs,output_hidden_states=True)
            results = outputs.entity_logits
            top_ten = torch.topk(results,k=number_of_words).indices
            convert = np.vectorize(id_to_tok)
            indices = np.array(top_ten.to('cpu'))
            tokens = convert(indices)
            predicted_entities+=([[terrier_query(y) for y in x[0] if '[UNK]' not in y ] for x in tokens.tolist()])
        del luke,inputs,outputs
        gc.collect()
        torch.cuda.empty_cache()
        return predicted_entities
    
### Takes in input a Dataframe with the evaluation sentences, the kind of method to expand and the number of entities to generate
### and returns the same DF with a new column for the expansions

def generate_expansions(evaluation, method = 'numberbatch',target_column='raw_utterance', entity_number = 25):
    df = evaluation.copy()
    if method == 'numberbatch':
        nounchunks = extract_nounchunks(df[target_column].tolist())
        entities = top_k_similar_from_nounchunks(nounchunks,k=entity_number)
        df ['expansion'] = entities
        return df
    if method == 'luke':
        luke = LukeEmbeddings('masked_large')
        entities = luke.generate_related_entities(df.raw_utterance.tolist(), number_of_words = entity_number)
        df['expansion'] = entities
        return df
    
def generate_expansions_new_knowledge(evaluation,method = 'numberbatch', entity_number = 25, use_context=False):        
    if method == 'numberbatch':
        if type(evaluation) ==str :
            nounchunks = extract_nounchunks([evaluation])
            entities = top_k_similar_from_nounchunks(nounchunks,k=entity_number)
            return entities
        else:
            nounchunks = extract_nounchunks(evaluation['raw_utterance'].tolist())
            entities = top_k_similar_from_nounchunks(nounchunks,k=entity_number)
            evaluation ['expansions'] = entities
            return evaluation
    if method == 'luke':
        luke = LukeEmbeddings('masked_large')
        if type(evaluation) ==str :
            entities = luke.generate_related_entities([evaluation], number_of_words = entity_number)
            return entities
        else:
            entities = luke.generate_related_entities(evaluation.raw_utterance.tolist(), number_of_words = entity_number)
            evaluation['expansions'] = entities
            return evaluation    
            
def generate_expansions_new_new(evaluation,method = 'numberbatch', entity_number = 25, use_context=False):        
    if method == 'numberbatch':
        if type(evaluation) ==str :
            nounchunks = extract_nounchunks([evaluation])
            entities = top_k_similar_from_nounchunks(nounchunks,k=entity_number)
            return entities
        else:
            nounchunks = extract_nounchunks(evaluation['raw_utterance'].tolist())
            entities = top_k_similar_from_nounchunks(nounchunks,k=entity_number)
            evaluation ['expansions'] = entities
            return evaluation
    if method == 'luke':
        luke = LukeEmbeddings('masked_large')
        if type(evaluation) ==str :
            entities = luke.generate_related_entities([evaluation], number_of_words = entity_number)
            return entities
        else:
            entities = luke.generate_related_entities(evaluation.raw_utterance.tolist(), number_of_words = entity_number)
            evaluation['expansions'] = entities
            return evaluation    
        
def generate_expansions_new(evaluation,method = 'numberbatch', entity_number = 25):        
    if method == 'numberbatch':
        if type(evaluation) ==str :
            nounchunks = extract_nounchunks([evaluation])
            entities = top_k_similar_from_nounchunks(nounchunks,k=entity_number)
            return entities
        else:
            nounchunks = extract_nounchunks(evaluation['raw_utterance'].tolist())
            entities = top_k_similar_from_nounchunks(nounchunks,k=entity_number)
            evaluation ['expansions'] = entities
            return evaluation
    if method == 'luke':
        luke = LukeEmbeddings('masked_large')
        if type(evaluation) ==str :
            entities = luke.generate_related_entities([evaluation], number_of_words = entity_number)
            return entities
        else:
            entities = luke.generate_related_entities(evaluation.raw_utterance.tolist(), number_of_words = entity_number)
            evaluation['expansions'] = entities
            return evaluation
        










### Generate the context giver the column target and the numebr of previous turns to consider
### if n_turns == 0 then it takes the whole previous context
def generate_context(df, column_target = 'raw_utterance',n_turns =3):
    all_context = []
    if 'conv_id' not in df.columns:
        df['conv_id'] = [x.split('_')[0] for x in df.qid.tolist()]
    if 'turn' not in df.columns:
        df['turn'] = [x.split('_')[1] for x in df.qid.tolist()]
    contexts = df.groupby('conv_id')[column_target].apply(list)
    for conv_id in df.conv_id.unique():
        sentences = contexts[conv_id]
        for i in range(len(sentences)):
            if i == 0: all_context.append('')
            elif i < n_turns: all_context.append(" ".join(sentences[:i]))
            else:
                if n_turns == 0:  
                    all_context.append(" ".join(sentences[:i]))
                else:
                    all_context.append(" ".join(sentences[i-n_turns:i]))
    return all_context


#%%
from nltk.stem import PorterStemmer

def generate_expansions_not_in_context(df, k=5,  text_column = 'raw_utterance', context_column = 'context' ):
    ps = PorterStemmer()
    embedding_matrix = torch.load('../expansion generator/numberbatch/numberbatch_embedding_matrix.pt')
    embedding_matrix = embedding_matrix.to('cuda')
    dictionary_numberbatch = load_dict('../expansion generator/numberbatch/numberbatch_dictionary.pkl')
    inverted_dict = {i:x for i,x in enumerate(list(dictionary_numberbatch.keys()))}
    assert text_column in df.columns and context_column in df.columns, 'The columns you indicated are not in the DF index, you can set text_column = columns where the utterancese are and context column the context column'
    expansions = []
    for i in tqdm(range(len(df))):
        noun_chunks_utterance = extract_nounchunks_from_str(df.raw_utterance.iloc[i])
        stemmed_context =[ps.stem(x) for x in  (df.context.iloc[i] + ' ' + df.raw_utterance.iloc[i]).split()]
        expansions_per_row = []
        for noun_chunk in noun_chunks_utterance:
            noun_chunk_ = noun_chunk.replace(' ','_')
            if noun_chunk_ in dictionary_numberbatch.keys():
                ind = dictionary_numberbatch[noun_chunk_]
                embedding = embedding_matrix[ind]
                dist = cosine_similarity(embedding_matrix,embedding.unsqueeze(dim=0))
                top_k_indices = torch.topk(dist,k+5).indices.to('cpu')
                top_k_words = [inverted_dict[int(x)] for x in top_k_indices]
                new_expansions = []
                counter = 0
                for top_word in top_k_words:
                    isin = False
                    splitted_stemmed = [ps.stem(x) for x in top_word.split('_')]
                    for splitted  in splitted_stemmed:
                        if splitted in stemmed_context:
                            isin = True
                    if not(isin):
                        if counter >= k:
                            continue
                        else:
                            new_expansions.append(top_word.replace('_', ' '))
                            counter += 1
                expansions_per_row += (new_expansions)
        expansions.append(expansions_per_row)
    return expansions    


