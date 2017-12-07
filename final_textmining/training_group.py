import parser
import pickle
import numpy
from sklearn.feature_extraction import DictVectorizer

def unlabelslist(n):
    list_unlabels = [-1] * n
    return(list_unlabels)

def get_labels(index_words):
    with open('lexicon.pickle', 'rb') as handle:
        lexicon = pickle.load(handle)
    #ARMO LISTA DE LABELS
    n = len(index_words)
    labels = unlabelslist(n)
    for i, word in enumerate(index_words):
        polaridad = float(lexicon.get(word, -1))
        if polaridad < 0 and polaridad != float(-1):
            labels[i] = 0
        elif polaridad > 0:
            labels[i] = 1
    return(labels)

def group_1(lexicon):
    with open('preprocessed_unlabeling.pickle', 'rb') as handle:
        sentences = pickle.load(handle)
    wordsDic, index_words, list_context = build_training_group1(sentences, lexicon, 1)
    v = DictVectorizer(sparse=True)
    X = v.fit_transform(list_context)
    labels = get_labels(index_words)
    #reduccion de dimensionalidad
    return(X, labels, wordsDic, index_words, list_context)

def group_2(lexicon):
    with open('preprocessed_unlabeling.pickle', 'rb') as handle:
        sentences = pickle.load(handle)
    wordsDic, index_words, list_context = build_training_group2(sentences, lexicon, 1)

    prom_contextos = 0
    count_context = 0
    for context in list_context:
        count_context += len(context)
    prom_contextos = count_context/len(list_context)

    wordsDic, index_words, list_context = reduce_data2(wordsDic, index_words, list_context, prom_contextos)


    return(wordsDic, index_words, list_context)  

def reduce_data2(wordsDic, index_words, list_context, prom_contextos):
    count_context = []
    wordsDic_new = {}
    index_words_new = []
    list_context_new = []
    for context in list_context:
        count_context.append(len(context))
    for i,context in enumerate(count_context):
        if context > prom_contextos:
            len_dic = len(wordsDic_new)
            wordsDic_new[index_words[i]] = len_dic
            index_words_new.append(index_words[i])
            list_context_new.append(list_context[i]) 
    return(wordsDic_new, index_words_new, list_context_new)
    
    
def build_training_group1(sentences,lexicon, option_context=1):
    wordsDic = {}
    index_words = []
    list_context = []
    if option_context == 1:
        for sent in sentences:
            for i in range(len(sent)):
                word = sent[i]
                word_before = w_before(sent,i)
                word_after = w_after(sent,i)

                pol_before = float(lexicon.get(word_before, 0))
                pol_after = float(lexicon.get(word_after, 0))
                if  pol_before != 0 or pol_after  != 0:
                    if wordsDic.get(word, None) == None:
                        lenDic = len(wordsDic)
                        wordsDic[word] = lenDic
                        index_words.append(word)
                        if pol_after != 0 and pol_before !=0:
                            list_context.append({word_after : 1, word_before : 1})
                        elif pol_before != 0:
                            list_context.append({word_before : 1})
                        else:
                            list_context.append({word_after : 1})
                    else:
                        index = wordsDic[word]
                        if pol_before != 0:
                            #si no esta en el contexto
                            if list_context[index].get(word_before,None) == None:
                                list_context[index][word_before] = 1
                            else:
                                list_context[index][word_before] += 1
                        if pol_after != 0:
                            #si no esta en el contexto                                
                            if list_context[index].get(word_after,None) == None:
                                list_context[index][word_after] = 1
                            else:
                                list_context[index][word_after] += 1
    wordsDic, index_words, list_context = reduce_data1(wordsDic, index_words, list_context, 4)
    return(wordsDic, index_words, list_context)

def reduce_data1(wordsDic, index_words, list_context, min_aparicion):
    count_aparitions = []
    count_context = []
    wordsDic_new = {}
    index_words_new = []
    list_context_new = []
    for context in list_context:
        count_context.append(len(context))
        count_aparitions.append(sum(context.values()))
    for i,context in enumerate(count_context):
        if context > min_aparicion:
            len_dic = len(wordsDic_new)
            wordsDic_new[index_words[i]] = len_dic
            index_words_new.append(index_words[i])
            list_context_new.append(list_context[i])         
    return(wordsDic_new, index_words_new, list_context_new)



def build_training_group2(sentences, lexicon, option_context=1):
    wordsDic = {}
    index_words = []
    list_context = []
    if option_context == 1:
        for sent in sentences:
            for i in range(len(sent)):
                word = sent[i]
                word_before = w_before(sent,i)
                word_after = w_after(sent,i)

                pol_before = float(lexicon.get(word_before, 0))
                pol_after = float(lexicon.get(word_after, 0))
                if  pol_before != 0 or pol_after  != 0:
                    if wordsDic.get(word, None) == None:
                        lenDic = len(wordsDic)
                        wordsDic[word] = lenDic
                        index_words.append(word)
                        if pol_after != 0 and pol_before !=0:
                            list_context.append({word_after : 1, word_before : 1})
                        elif pol_before != 0:
                            list_context.append({word_before : 1})
                        else:
                            list_context.append({word_after : 1})
                    else:
                        index = wordsDic[word]
                        if pol_before != 0:
                            #si no esta en el contexto
                            if list_context[index].get(word_before,None) == None:
                                list_context[index][word_before] = 1
                            else:
                                list_context[index][word_before] += 1
                        if pol_after != 0:
                            #si no esta en el contexto                                
                            if list_context[index].get(word_after,None) == None:
                                list_context[index][word_after] = 1
                            else:
                                list_context[index][word_after] += 1
    return(wordsDic, index_words, list_context)

def get_values(a,b):
    positive = 0
    negative = 0
    if a < 0:
        negative += 1
    elif a > 0:
        positive += 1
    if b < 0:
        negative += 1
    elif b > 0:
        positive += 1
    return(positive, negative)    

def w_before(sent, i):
    if i!= 0:
        return sent[i-1]
    else:
        return "[start]"

def w_after(sent, i):
    if i < len(sent)-1:
        return sent[i+1]
    else:
        return "[end]"