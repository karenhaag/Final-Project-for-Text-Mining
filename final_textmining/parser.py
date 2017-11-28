import xml.etree.ElementTree as ET
import pickle
import string

##with open('positiveWords.pickle', 'rb') as handle:
##    b = pickle.load(handle)



def get_semillas(path='senticon.es.xml'):
    tree = ET.parse(path)
    root = tree.getroot()
    positiveWords = {}
    negativeWords = {}
    for positive in root.iter('positive'):
        for lemma in positive:
            #con strip le saco espacios
            positiveWords[lemma.text.strip()] = lemma.get("pol")
    for negative in root.iter('negative'):
        for lemma in negative:
            negativeWords[lemma.text.strip()] = lemma.get("pol")
    
    with open('positiveWords.pickle', 'wb') as handle:
        pickle.dump(positiveWords, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("positive words saved")
    with open('negativeWords.pickle', 'wb') as handle:
        pickle.dump(negativeWords, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("negative words saved")
    

def cleanTweetCorpus(path="general-tweets-test.xml"):
    #general-tweets-test1k.xml
    tree = ET.parse(path)
    root = tree.getroot()
    tweets = []
    for data in root.iter("content"):
        tweets.append(data.text)
    return(tweets)





#limpia el corpus de tweeters de hashtag, mensiones, signos de puntuación, etc. 
#tambien pasa las palabras a lowercase
def preproces_tweets():
    exclude = string.punctuation
    exclude = exclude + "?" + "”" + "¡" + "¿"
    prefixes = ('http')
    tweets = cleanTweetCorpus()
    aux = []
    for tweet in tweets:
        aux.append(tweet.split())
    list_clean_tweets = []
    for sentence in aux:
        sentence =  [x for x in sentence if not any(c.isdigit() for c in x)]
        sentence =  [x for x in sentence if not any(c == "@" for c in x)]
        sentence =  [x for x in sentence if not any(c == "#" for c in x)]
        sentence =  [x for x in sentence if not any(c == "\\" for c in x)]
        sentence =  [x for x in sentence if not x.startswith(prefixes)]
        sen = []
        for word in sentence:
            s = ''.join(ch for ch in word if ch not in exclude)
            if s != '':
                sen.append(s.lower())
        list_clean_tweets.append(sen)
    return(list_clean_tweets)

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


#return a dic with (key=word, value=index in index_words), a index_words list and
# a list of context for each word
def build_training_group(sentences, option_context=1):
    wordsDic = {}
    index_words = []
    list_context = []
    if option_context == 1:
        for sent in sentences:
            for i in range(len(sent)):
                word = sent[i]
                word_before = w_before(sent,i)
                word_after = w_after(sent,i)
                #armado wordsDic
                if wordsDic.get(word, None) == None:
                    lenDic = len(wordsDic)
                    wordsDic[word] = lenDic
                    index_words.append(word)
                    list_context.append({"w_after"+ word_after : 1, "w_before"+ word_before: 1})
                else:
                    index = wordsDic[word]
                    context_before = "w_before"+ word_before
                    context_after = "w_after"+ word_after
                    #si no esta en el contexto
                    if list_context[index].get(context_before,None) == None:
                        list_context[index][context_before] = 1
                    else:
                        list_context[index][context_before] += 1
                    if list_context[index].get(context_after,None) == None:
                        list_context[index][context_after] = 1
                    else:
                        list_context[index][context_after] += 1
    return(wordsDic, index_words, list_context)


""" 
with open('positiveWords.pickle', 'rb') as handle:
        b = pickle.load(handle)
"""