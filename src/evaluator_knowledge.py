import time
import collections
import numpy
import numpy as np
import random
import ast
import math
from random import randint
import codecs
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class MLTEvaluator(object):
    def __init__(self, config):
        self.config = config
        self.cost_sum = 0.0
        self.sentence_predicted=[]
        self.sentence_correct=[] 
        self.sentence_total = []
        self.X_test = []
        self.token_scores_list=[]
        self.start_time = time.time()
        self.count=0
        self.prec=0
        self.tot=0
        self.rec=0
        self.pos=0

    def append_data(self, cost, batch, sentence_scores, token_scores_s, token_scores_t, name, epoch):
        assert(len(batch) == len(sentence_scores))
        self.cost_sum += cost
        sentence_pred_refined=[]
        sentence_cor_refined=[]
        id,sentences,target_sentences,source_sentiment, target_sentiment, knowledge,label_distribution = zip(*batch)
        max_1=0
        X = sentences
        y = label_distribution
        for i in range(len(X)):
            sentence_s=[]
            self.token_scores_list.append(token_scores_s)
            sentence_cor = []
            sentence_pred = []
            a = y[i]
            pos = [i for i, j in enumerate(a) if j == 1]
            if pos[0] == 0:
                    sentence_cor.append(1)
            else: 
                    sentence_cor.append(0)
            pos_pred = []
            if sentence_scores[i][0]>=sentence_scores[i][1]: 
                   sentence_pred.append(1)    
            else: 
                   sentence_pred.append(0) 
            if name=='test':
                      f1  = codecs.open(self.config["output_path"]+"result_"+str(epoch)+".txt",encoding='utf-8',mode='a')
                      s = str(id[i])+'\t'+str(X[i])+'\t'+str(target_sentences[i]) +'\t'+ str(y[i])+'\t'+ str(sentence_scores[i])+ '\n'
                      f1.write(s)
                      #s = str(id[i])+'\t'+str(token_scores_s[i])+'\n'
                      #f1.write(s)
                      #s = str(id[i])+'\t'+str(token_scores_t[i])+'\n'
                      #f1.write(s)
                    
            self.sentence_predicted.append(sentence_pred[-1])
            self.sentence_correct.append(sentence_cor[-1])
                
    def get_results(self, name):
        #assert(len(self.sentence_correct[0])==len(self.sentence_predicted[0]))
        print("GETTING RESULTs")
        f=[]
        p=[]
        r=[]
        acc=[]
        show={}
        f1_macro = 0
        p1_macro = 0
        r1_macro = 0
        f2_weighted = 0
        p2_weighted = 0
        r2_weighted = 0
        f1_w = 0
        p2 = precision_score(np.array(self.sentence_correct), np.array(self.sentence_predicted),average='macro')      
        r2 = recall_score(np.array(self.sentence_correct), np.array(self.sentence_predicted),average='macro')
        f1_macro = f1_score(np.array(self.sentence_correct), np.array(self.sentence_predicted),average='macro')
        f1_w = f1_score(np.array(self.sentence_correct),np.array(self.sentence_predicted),average="weighted")
        con = confusion_matrix(np.array(self.sentence_correct), np.array(self.sentence_predicted))
        print(con)
        macrof1 = 2*p2*r2
        macrof1/= (p2+r2)
        f2 = macrof1
        print(len(self.sentence_correct))  
        results = collections.OrderedDict()
        results[name + "_cost_sum"] = self.cost_sum
        results[name + "_tok_"+str()+"_p"] = p2*100
        results[name + "_tok_"+str()+"_r"] = r2*100
        results[name + "_tok_"+str()+"_f_harmonic"] = f2*100
        #results[name + "_tok_"+str()+"_f_weight"] = f1_w*100
        results[name + "_tok_"+str()+"confusion"] =con
        results[name + "_tok_"+str()+"_f"] = f1_macro*100
        
        results[name + "_time"] = float(time.time()) - float(self.start_time)

        return results
