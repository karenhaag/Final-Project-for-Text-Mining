import xml.etree.ElementTree as ET
import pickle
import string

##with open('lexicon.pickle', 'rb') as handle:
##    b = pickle.load(handle)


##TAREAS:
#eliminar stopwords en preprocess tweets



def tweetCorpusLabeled(path='general-tweets-train-tagged.xml'):
    tree = ET.parse(path)
    root = tree.getroot()
    tweets = []
    for data in root.iter("content"):
        tweets.append(data.text)  
    polarity_tweets = []
    for data in root.iter("polarity"):
        if data.find("entity")== None:
            pol = data.find("value").text
            polarity_tweets.append(pol)
    return(tweets, polarity_tweets)

def tweetCorpusUnlabeled(path="general-tweets-test.xml"):
    tree = ET.parse(path)
    root = tree.getroot()
    tweets = []
    for data in root.iter("content"):
        tweets.append(data.text)
    return(tweets)

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
    #saco las palabras que estan en ambas listas
    repeated_words=[]
    for word in positiveWords:
        if negativeWords.get(word, None) != None:
            repeated_words.append(word)
    for word in repeated_words:
        """
        polarity = get_polarity(word)
        if polarity < 0:
            positiveWords.pop(word, None)
        elif polarity > 0:
            negativeWords.pop(word, None)
        else:
        """
        if -float(negativeWords.get(word)) < float(positiveWords.get(word)):
            negativeWords.pop(word, None)
        else:    
            positiveWords.pop(word, None)
    positiveWords.update(negativeWords)
    semillas = positiveWords
    with open('lexicon.pickle', 'wb') as handle:
        pickle.dump(semillas, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("lexicon saved")
            
        
    
def get_stopwords(words_file = "stopwords-es.txt"):
    stopwords =  [word for line in open(words_file, 'r') for word in line.split()]
    #sacamos de las stopwords las palabras que se encuentran en el lexicon
    with open('lexicon.pickle', 'rb') as handle:
        lexicon = pickle.load(handle)
    for word in stopwords:
         if lexicon.get(word, "None") != "None":
             stopwords.remove(word)
    
    stopwords.remove("bien")
    stopwords.remove("buen")
    stopwords.remove("vez")
    with open('stopwords.pickle', 'wb') as handle:
        pickle.dump(stopwords, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("stopwords saved")

def get_polarity(word):
    tweets, polarity_tweets = tweetCorpusLabeled()
    tweets = preproces_tweets(tweets)
    positive_tweets = 0
    negative_tweets = 0
    for tweet in tweets:
        if word in tweet:
            index = tweets.index(tweet)
            polarity = polarity_tweets[index]
            if polarity == "P" or polarity =="P+":
                positive_tweets += 1
            elif polarity == "N" or polarity =="N+":
                negative_tweets += 1
    if negative_tweets < positive_tweets:
        pol = 1
    elif negative_tweets > positive_tweets:
        pol = -1
    else:
        pol = 0
    return(pol)
                
#limpia el corpus de tweeters de hashtag, mensiones, signos de puntuación, etc. 
#tambien pasa las palabras a lowercase
#recibe una lista de tweeters
def preproces_tweets(tweets):
    with open('stopwords.pickle', 'rb') as handle:
        stopwords = pickle.load(handle)
    print("stop words load")
    stopwords = set(stopwords)
    """
    with open('lexicon.pickle', 'rb') as handle:
        lexicon = pickle.load(handle)
    print("lexicon cargado") 
    """
    exclude = string.punctuation
    exclude = exclude + "?" + "”" + "¡" + "¿"
    prefixes = ('http')
    aux = []
    for tweet in tweets:
        if type(tweet) == str:
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
                s = s.lower()
                if s not in stopwords:
                    sen.append(s)
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
def build_training_group(option_context=1):
    with open('preprocessed_unlabeling.pickle', 'rb') as handle:
        sentences = pickle.load(handle)
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