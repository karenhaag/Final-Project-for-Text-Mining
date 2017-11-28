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





#limpia el corpus de tweeters de hashtag, mensiones, signos de puntuación, etc
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


#return a list with all words of a sentence
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


def feature_selection(dic_context, ocurrences):
    for context_of_word in dic_contexts:
        
        for context in context_of_word:
            if context_of_word.get(context) < ocurrences:
                co




def prop1():
    """
    - propagación 1:
    - sobre un corpus, armamos un grafo de co-ocurrencias de las palabras, donde los nodos son las palabras y 
    existe un arco entre dos nodos si las palabras co-ocurren en un contexto k más de n veces. El arco no es 
    dirigido y los arcos tienen peso. El peso es alguna función del número de co-ocurrencias de las palabras 
    (número, tf*idf, IM, etc)

    - colocamos las etiquetas de sentimiento sobre este grafo,
    y las propagamos con http://scikit-learn.org/stable/modules/label_propagation.html, de forma que las 
    palabras queden asociadas a clases positivo, negativo y posiblemente neutro (por ejemplo, asociamos una
    palabra a la etiqueta "neutro" si queda asociada a "positivo" o "negativo" con una intensidad menor a r)
    """



def prop2():
    """
    - propagación 2:
    - vemos en un corpus las palabras que están más fuertemente asociadas a palabras de sentimiento positivo o negativo, 
    mediante Información Mútua, porque ocurren en un contexto k de la palabra semilla.
    - añadimos a las palabras semilla aquellas que tengan una fuerte Información Mútua
    - iteramos
    """
    sentences = preproces_tweets()
    wordsDic, index_words, list_context = build_training_group(sentences, 1)
    with open('positiveWords.pickle', 'rb') as handle:
        positiveDic = pickle.load(handle)
    with open('negativeWords.pickle', 'rb') as handle:
        negativeDic = pickle.load(handle)
    list_context = feature_selection(list_context)
    return(wordsDic, index_words, list_context)


def prop3():
    """
    - propagación 3:
    - para cada tweet, le asigno clase positiva o negativa usando el clasificador c con el lexicón inicial (semilla)
    - si un tweet no tiene ninguna palabra ni positiva ni negativa, lo descartamos como ejemplo de aprendizaje
    - incorporo las palabras que han sido asignadas a oraciones positivas como positivas (valor:1) y a oraciones negativas como negativas (valor: -1). Si una palabra ha sido asignada a ambas clases, le asigno un valor contínuo entre 1 y -1 que sea una función de la proporción de veces que ocurrió en oraciones positivas o negativas
    - usamos el clasificador con el lexicón aumentado con nuevas palabras para anotar nuevos ejemplos, iteramos
    """


""" 
with open('positiveWords.pickle', 'rb') as handle:
        b = pickle.load(handle)
"""