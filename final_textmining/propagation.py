import training_group
from sklearn.semi_supervised import LabelPropagation
import pickle
import parser
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score



def prop1(lexicon):
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

    data, labels, wordsDic, index_words, list_context = training_group.group_1(lexicon)
    data = data.toarray()
    label_prop_model = LabelPropagation()

    label_prop_model.fit(data, labels)
    label_distribution = label_prop_model.label_distributions_
    l = []
    for i,a in enumerate(label_distribution):
         if not np.isnan(a[0]) or not np.isnan(a[1]):
             l.append(i)
    #l tiene todos los elementos con sentimiento propagado
    new_lexicon = lexicon
    for i in l:
        print(index_words[i])
        if lexicon.get(index_words[i], None) == None:
            if label_distribution[i][0] > 0:
                new_lexicon[index_words[i]] = 0
            else :
                new_lexicon[index_words[i]] = 1
    return(new_lexicon)
                


    return(label_prop_model)



def prop2(lexicon):
    """
    - propagación 2:
    - vemos en un corpus las palabras que están más fuertemente asociadas a palabras de sentimiento positivo o negativo, 
    mediante Información Mútua, porque ocurren en un contexto k de la palabra semilla.
    - añadimos a las palabras semilla aquellas que tengan una fuerte Información Mútua
    - iteramos
    """
    """
    with open('lexicon.pickle', 'rb') as handle:
        lexicon = pickle.load(handle)
    """
    wordsDic, index_words, list_context = training_group.group_2(lexicon)

    new_lexicon = lexicon

    for i,context in enumerate(list_context):
        count_contPos = 0
        count_contNeg = 0 
        for key in context:
            if float(lexicon.get(key)) < 0.0:
                count_contNeg += 1
            else:
                count_contPos += 1
        if count_contNeg/(count_contNeg+count_contPos) > 0.7:
            if new_lexicon.get(index_words[i], None) == None:
                new_lexicon[index_words[i]] = float(-1)
        elif count_contPos/(count_contNeg+count_contPos) > 0.7:
            if new_lexicon.get(index_words[i], None) == None:
                new_lexicon[index_words[i]] = float(1)

    return(new_lexicon)
            

    #ver resultados clasificador con lexicon nuevo
    #volver a entrenar
    #conseguir otro corpuuus   
    #ver que pasa si le asigno positivo y negativo         


def prop3(lexicon):
    """
    - propagación 3:
    - para cada tweet, le asigno clase positiva o negativa usando el clasificador c con el lexicón inicial (semilla)
    - si un tweet no tiene ninguna palabra ni positiva ni negativa, lo descartamos como ejemplo de aprendizaje
    - incorporo las palabras que han sido asignadas a oraciones positivas como positivas (valor:1) y a oraciones negativas como 
    negativas (valor: -1). Si una palabra ha sido asignada a ambas clases, le asigno un valor contínuo entre 1 y -1 que sea una 
    función de la proporción de veces que ocurrió en oraciones positivas o negativas
    - usamos el clasificador con el lexicón aumentado con nuevas palabras para anotar nuevos ejemplos, iteramos
    """
    with open('preprocessed_unlabeling.pickle', 'rb') as handle:
        sentences = pickle.load(handle)
   
    for sent in sentences:
        polarity = clasificador_sentence(sent, lexicon)

        if polarity != "NEU":
            if polarity == "P" :
                for word in sent:
                    if lexicon.get(word, "None") == "None":
                        lexicon[word] = 1
            else:
                for word in sent:
                    if lexicon.get(word, "None") == "None":
                        lexicon[word] = -1
    return(lexicon)



def clasificador_sentence(sentence, lexicon):
    polarity_sentence = "NEU"
    count = 0
    for word in sentence:
        polarity = lexicon.get(word, "None")
        if polarity !=  "None":
            polarity = float(polarity)
            if polarity < 0.0:
                count -= 1 
            if polarity > 0.0:
                count += 1
    if count > 0:
        polarity_sentence = "P"
    elif  count < 0:
        polarity_sentence = "N"
    return(polarity_sentence)

def get_real_polarity(real_polarity):
    if real_polarity == "N+":
        real_polarity = "N"
    elif real_polarity == "P+":
        real_polarity = "P"
    return(real_polarity)



def evaluator(propagation, iteration):
    predicciones = []
    with open('lexicon.pickle', 'rb') as handle:
        lexicon = pickle.load(handle)
    sentences, polarity_sentences = parser.tweetCorpusLabeled()
    for i in range(0, iteration):
        if i == 0:
            new_lexicon = lexicon
        else:
            if propagation == 1:
                new_lexicon = prop1(lexicon)
            elif propagation == 2:
                new_lexicon = prop2(lexicon)
            elif propagation == 3:
                new_lexicon = prop3(lexicon)

        for j,sent in enumerate(sentences):
            real_polarity = polarity_sentences[j]
            
            if real_polarity != "NONE" and real_polarity != "NEU":
                pol_clasificador = clasificador_sentence(sent, new_lexicon)
                real_polarity = get_real_polarity(real_polarity)
                if real_polarity == pol_clasificador:
                    if real_polarity == "P":
                        real_polarity_2 = 1
                        pol_clasificador_2 = 1
                    else:
                        real_polarity_2 = -1
                        pol_clasificador_2 = -1
                else:
                    if real_polarity == "P":
                        real_polarity_2 = 1
                        pol_clasificador_2 = -1
                    else:
                        real_polarity_2 = -1
                        pol_clasificador_2 = 1
                predicciones.append([i, real_polarity_2, pol_clasificador_2])
        lexicon = new_lexicon
    predictions_df = pd.DataFrame(predicciones, columns=["iteration", "true", "pred"])
    if propagation == 1:
        predictions_df.to_csv('./data1.csv', index=False)
    if propagation == 2:
        predictions_df.to_csv('./data2.csv', index=False)
    if propagation == 3:
        predictions_df.to_csv('./data3.csv', index=False)
        
    
        

def calculate_metrics(df):
    rdf = {}
    rdf['precision'] = precision_score(df.true, df.pred, average='macro')
    rdf['recall'] = recall_score(df.true, df.pred, average='macro')    
    return pd.Series(rdf)  
"""

En una tarea de clasificación, un puntaje de precisión de 1.0 para una clase 
C significa que cada elemento etiquetado como perteneciente a la clase C 
pertenece a la clase C (pero no dice nada sobre el número de elementos 
de la clase C que no fueron etiquetados correctamente) mientras que el recuerdo 
de 1.0 significa que cada elemento de la clase C se etiquetó como 
perteneciente a la clase C (pero no dice nada sobre cuántos otros elementos 
también se etiquetaron incorrectamente como pertenecientes a la clase C).
"""