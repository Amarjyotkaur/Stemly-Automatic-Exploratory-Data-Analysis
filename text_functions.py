# Text Functions

import nltk

from nltk.tokenize import word_tokenize
import pandas as pd
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
from bs4 import BeautifulSoup

import numpy as np
from nltk.tokenize.toktok import ToktokTokenizer
import re

import unicodedata
from nltk.corpus import brown


tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer 
nltk.download('brown') 
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp=en_core_web_sm.load()

import re
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models.coherencemodel import CoherenceModel
#from gensim.models import Cohere
# spacy for lemmatizationnceModel

import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt


# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

"""
Text Preprocessing / Cleaning
1. Remove Stop Words
2. Tokenize
3. Strip HTML Tags
4. Remove Accented Chars
5. Lemmatization
6. Remove Special Characters
7. Expand Contractions"""

def remove_stopwords_col(colData):
    """Remove Stop Words
    Returns list of input text without stop words
    Args: 
     colData (array_like, 1D):Pandas Series of Data or Dataframe Column of text data for function to be applied on
    
    Returns:
     ls : list of String texts without stop words, for every row in input column
    
    Example: 
    >>ft.remove_stopwords_col(pd.Series(["hello my name is john","this is an apple")
    ['hello name john', 'apple']
     
    """
    stop_words =set(stopwords.words('english'))
    ls=[]
    for s in colData:
        tokens = [w for w in s if not w in stop_words]
        ls.append(tokens)
    return ls

def tokenize_column(colData):
    """Tokenize Column
    Returns list of lists of tokenized words from input column of text
    Args: 
     colData (array_like, 1D):Pandas Series of Data or Dataframe Column of text data for function to be applied on
    
    Returns:
     ls : list of lists of tokenized String words
    
    Example: 
    >>ft.tokenize_column(pd.Series(["hello my name is john", "this is an apple"]))
    [['hello', 'my', 'name', 'is', 'john'], ['this', 'is', 'an', 'apple']]
     
    """
    ls=[]
    for s in colData:
        tokens = word_tokenize(s)
        ls.append(tokens)
    return ls

def tokenize_str(text):
    """Tokenize String
    Returns list of lists of tokenized words from input String
    Args: 
     text : String
    
    Returns:
     tokens : list of tokenised words from input string 
    
    Example: 
    >>ft.tokenize_str('hello my name is john')
    ['hello', 'my', 'name', 'is', 'john']
     
    """
    tokens = word_tokenize(text)
    return tokens


def strip_html_col(colData):
    """Strip HTML Tags in Column
    Returns list of lists of input text without HTML tags
    Args: 
     colData (array_like, 1D):Pandas Series of Data or Dataframe Column of text Data for function to be applied on
    
    Returns:
     ls : list of String input text without HTML Tags
    
    Example: 
    >>ft.strip_html_col(pd.Series(["<h>hello my name is john<h/>", "this is an apple"]))
    ['hello my name is john', 'this is an apple']
     
    """
    ls=[]
    for i in colData:
        soup = BeautifulSoup(i, "html.parser")
        stripped_text = soup.get_text()
        ls.append(stripped_text)
    return ls


def strip_html_str(text):
    """Strip HTML Tags in String
    Returns String input text without HTML tags
    Args: 
     text : String
    Returns:
     String input text without HTML Tags
    
    Example: 
    >>ft.strip_html_str("<h>hello my name is john<h/>")
    'hello my name is john'
     
    """
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text()
    return stripped_text

def remove_accented_chars(text):
    """Remove Accented Characters 
    Returns String input text without accented characters
    Args: 
     text : String
    Returns:
     String input text without accented characters
    
    Example: 
    >>ft.remove_accented_chars("Helloç my name is john")
    'helloc my name is john'
     
    """
    
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text


def remove_special_characters(text):
    """Remove Accented Characters 
    Returns String input text without special characters
    Args: 
     text : String
    Returns:
     st: String input text without special characters
    
    Example: 
    >>ft.remove_special_characters("Helloç my name is john~\-=")
    'Hello my name is john\\'
     
    """
    text = re.sub('[^a-zA-z0-9\s]', '', text)
    return text

def lemmatize_text(text):
    """Lemmatize text
    Returns String input text with lemmatized words
    Args: 
     text : String
    Returns:
     st: String input text with lemmatized words
    
    Example: 
    >>ft.lemmatize_text("hello these are my cats, they are eating oranges.")
    ' hello these are my cat , they are eating orange .'
    """
     
    # Init the Wordnet Lemmatizer
    text = tokenize_str(text)
    st =""
    lemmatizer = WordNetLemmatizer()
    for w in text:
        #print(w)
        new_w = lemmatizer.lemmatize(w)
        st = st+" "+new_w
    return st


def remove_stopwords_str(text, is_lower_case=False):
    """Remove Stop Words String
    Returns input text without stop words
    Args: 
     text: String
    Returns:
     input String without stop words
    
    Example: 
    >>ft.remove_stopwords_str("hello my name is john")
    'hello name john'
     
    """
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text



def normalize_corpus(corpus, html_stripping=True, contraction_expansion=True,
                     accented_char_removal=True, text_lower_case=True, 
                     text_lemmatization=True, special_char_removal=True, 
                     stopword_removal=True, tokenize_text =True):
    """Normalize Corpus
    Returns Series of text with applied pre-processing functions
    Args: 
     corpus : Pandas Series of text Data or Dataframe Column of text Data for function to be applied on
    Returns:
      Series of normalized list of Strings
    
    Example: 
    >>ft.normalize_corpus(pd.Series(['hello! these are my birds#', 'they are eating oranges.']))
    0       [hello, bird]
    1    [eating, orange]
    dtype: object
    """
    
    normalized_corpus = []
    # normalize each document in the corpus
    for st in corpus:
        st = str(st)
        # strip HTML
        if html_stripping:
            st = strip_html_str(st)
        # remove accented characters
        if accented_char_removal:
            st = remove_accented_chars(st)
        # expand contractions    
       # if contraction_expansion:
        #    doc = expand_contractions(doc)
        # lowercase the text    
        if text_lower_case:
            st = st.lower()
        # remove extra newlines
        st = re.sub(r'[\r|\n|\r\n]+', ' ',st)
        # insert spaces between special characters to isolate them    
        special_char_pattern = re.compile(r'([{.(-)!}])')
        st = special_char_pattern.sub(" \\1 ", st)
        # remove stopwords
        if stopword_removal:
            st = remove_stopwords_str(st, is_lower_case=text_lower_case)
        # lemmatize text
        if text_lemmatization:
            st = lemmatize_text(st)
        # remove special characters    
        if special_char_removal:
            st = remove_special_characters(st)  
        # remove extra whitespace
        st = re.sub(' +', ' ', st)
        if tokenize_text:
            st = tokenize_str(st)
            
            
        normalized_corpus.append(st)
        
    return pd.Series(normalized_corpus)



"""
Basic feature extraction using text data
1. Number of words
2. Number of characters
3. No. of Stop Words,
4. Length of Words
5. Average Length of Words
6. Number of Digits
7. Number of Alphabets
8. Number of Spaces
9. Extracted Digits
10. List of Nouns
11. Number of Nouns
12. Extract Entity
13. Entity Count
14. Entity List
14. Part of Speech Tag
15. Part of Speech List
16. Part of Speech Count


Number of special characters,
Number of uppercase words,
Number of Lower case words"""


def words_c(colData):
    """Word Count
    Returns list of number of words for every row in input column
    Args: 
     colData (array_like, 1D):Pandas Series of Data or Dataframe Column of text Data for function to be applied on
    Returns:
     ls : list of number of words for every row in input column
    
    Example: 
    >>ft.words_c(pd.Series(['hello my name is john', 'these are oranges', 'the sky is blue']))
    [5, 3, 4]
     
    """
    ls=[] 
    for s in colData:
        s= str(s)
        #print(s)
        s= s.split()
        ls.append(len(s))
    return ls

def char_c(colData):
    """Character Count
    Returns list of number of characters for every row in input column
    Args: 
     colData (array_like, 1D):Pandas Series of Data or Dataframe Column of text Data for function to be applied on
    Returns:
     ls : list of number of characters for every row in input column
    
    Example: 
    >>ft.char_c(pd.Series(['hello my name is john', 'these are oranges', 'the sky is blue']))
    [21, 17, 15]
     
    """
    ls=[]
    for s in colData:
        s= str(s)
        #print(s)
        ls.append(len(s))
    return ls

def stopwords_c(colData):
    """Stopwords Count
    Returns list of number of stopwords for every row in input column
    Args: 
     colData (array_like, 1D):Pandas Series of Data or Dataframe Column of text Data for function to be applied on
    Returns:
     result : list of number of stopwords for every row in input column
    
    Example: 
    >>ft.stopwords_c(pd.Series(['hello my name is john', 'these are oranges', 'the sky is blue']))
    [2, 2, 2]
     
    """
    result = []
    for st in colData:
        count = 0
        tokens = tokenizer.tokenize(st)
        tokens = [token.strip() for token in tokens]
        for i in tokens:
            if i in stopword_list:
                count+=1
        result.append(count)
    return result
        
    
#helper function
def len_words(colData):
    """Length of Words
    Returns Series of lists of length of words for every row in input column
    Args: 
     colData (array_like, 1D):Pandas Series of Data or Dataframe Column of text Data for function to be applied on
    Returns:
     result : Series of lists of length of words for every row in input column
    
    Example: 
    >>ft.len_words(pd.Series(['hello my name is john', 'these are oranges', 'the sky is blue']))
    0    [5, 2, 4, 2, 4]
    1          [5, 3, 7]
    2       [3, 3, 2, 4]
    dtype: object
     
    """
    result = []
    for st in colData:
        ls=[]
        st = st.split()
        for w in st:
            ls.append(len(w))
        result.append(ls)
    return pd.Series(result)

def avg_length(colData):
    """Average Length of Words
    Returns list of average length of words per row in input column
    Args: 
     colData (array_like, 1D):Pandas Series of Data or Dataframe Column of text Data for function to be applied on
    Returns:
     result : list of float average word length
    
    Example: 
    >>ft.avg_length(pd.Series(['hello my name is john', 'these are oranges', 'the sky is blue']))
    [4.2, 5.666666666666667, 3.75]
     
    """
    result=[]
    for i in range(len(char_c(colData))):
        avg = char_c(colData)[i]/words_c(colData)[i]
        result.append(avg)
    return result

def digits_c(colData):
    """Digits Count
    Returns list of number of digits for every row in input column
    Args: 
     colData (array_like, 1D):Pandas Series of Data or Dataframe Column of text Data for function to be applied on
    Returns:
     ls : list of number of digits for every row in input column
    
    Example: 
    >>ft.digits_c(pd.Series(['hello my name is john', 'these are oranges', 'the sky is blue']))
    [0,0,0]
     
    """
    
    ls =[]
    for s in colData:
        s= str(s)
        count = 0
        for char in s:
            if char.isdigit():
                count+=1
        ls.append(count)
    return ls


def alph_c(colData):
    """Alphabet Count
    Returns list of number of alphabets for every row in input column
    Args: 
     colData (array_like, 1D):Pandas Series of Data or Dataframe Column of text Data for function to be applied on
    Returns:
     ls : list of number of alphabets for every row in input column
    
    Example: 
    >>ft.alph_c(pd.Series(['hello my name is john', 'these are oranges', 'the sky is blue']))
    [17, 15, 12]
     
    """
    
    ls=[]
    for s in colData:
        s = str(s)
        count = 0
        for char in s:
            char =str(char)
            if char.isalpha():
                count+=1
        ls.append(count)
    return ls



def space_c(colData):
    """Spaces Count
    Returns list of number of spaces for every row in input column
    Args: 
     colData (array_like, 1D):Pandas Series of Data or Dataframe Column of text Data for function to be applied on
    Returns:
     ls : list of number of spaces for every row in input column
    
    Example: 
    >>ft.space_c(pd.Series(['hello my name is john', 'these are oranges', 'the sky is blue']))
    [4, 2, 3]
     
    """
    ls = []
    for s in colData:
   
        count = 0
        for char in s:
            if char ==" ":
                count+=1
        ls.append(count)
    return ls



def extract_num_ls(colData):
    """Extracted Numbers List
    Returns list of lists of numbers (separated by spaces) in every row of input column
    Args: 
     colData (array_like, 1D):Pandas Series of Data or Dataframe Column of text Data for function to be applied on
    Returns:
     result : list of list of numbers in every row of input column
    Example: 
    >>ft.extract_num_ls(pd.Series(['hello my name is john', 'these are 20 oranges', 'the sky is blue']))
    [[], ['20'], []]
     
    """
    
    result = []
    for s in colData:
        tokens = tokenize_str(s)
        innerls=[]
        #print(tokens)
        for w in tokens:
            for c in w:
                if c.isdigit():
                    if w not in innerls:
                        innerls.append(w)
        result.append(innerls)
    return result


def num_c(colData):
    """Number Count
    Returns list of Numbers (separated by spaces) for every row in input column
    Args: 
     colData (array_like, 1D):Pandas Series of Data or Dataframe Column of text Data for function to be applied on
    Returns:
     result : list of count of numbers for every row in input column
    
    Example: 
    >>ft.num_c(pd.Series(['hello my name is john', 'these are 20 oranges', 'the sky is blue']))
    [0, 1, 0]
     
    """
    result = []
    for ls in extract_num_ls(colData):
        result.append(len(ls))
    return result
        
    
def nouns(colData):
    """Nouns
    Returns Series of Nouns in every row in input column
    Args: 
     colData (array_like, 1D):Pandas Series of Data or Dataframe Column of text Data for function to be applied on
    Returns:
     Series of nouns for every row in input column
    
    Example: 
    >>ft.nouns(pd.Series(['hello my name is john', 'these are oranges', 'the sky is blue']))
    0    [hello, name]
    1        [oranges]
    2            [sky]
    dtype: object

    """
    ls=[]
    for s in colData:
        sum(1 for word, pos in pos_tag(word_tokenize(s)) if pos.startswith('NN'))
        nouns = [word for word, pos in pos_tag(word_tokenize(s)) if pos.startswith('NN')]
        #nouns = [(word,pos) for word, pos in pos_tag(word_tokenize(s)) if pos.startswith('NN')]
        ls.append(nouns)
        
    return pd.Series(ls)



def num_noun(colData):
    """Number of Nouns
    Returns list of number of Nouns in every row in input column
    Args: 
     colData (array_like, 1D):Pandas Series of Data or Dataframe Column of text Data for function to be applied on
    Returns:
     result : list of integer count of nouns for every row in input column
    
    Example: 
    >>ft.num_noun(pd.Series(['hello my name is john', 'these are oranges', 'the sky is blue']))
    [2, 1, 1]
    """
    
    result=[]
    for ls in nouns(colData):
        result.append(len(ls))
    return result


def extract_entity(colData): # colData does not have to be preprocessed!
    """Extract Entity
    Returns list of all extracted entities in every row of input column
    Args: 
     colData (array_like, 1D):Pandas Series of Data or Dataframe Column of text Data for function to be applied on
    Returns:
     result : list of lists of extracted entities in every row in input column
    
    Example: 
    >>ft.extract_entity(pd.Series(['hello my name is john', 'these are oranges', 'the sky is blue']))
    [[('john', 'PERSON')], [], []]

    """
    result=[]
    for i in colData:
        doc = nlp(i)
        #esult.append([(X, X.ent_iob_, X.ent_type_) for X in doc])
        result.append([(X.text, X.label_) for X in doc.ents])
        
    return result



def entity_c(colData):
    """Entity Count
    Returns list of all extracted entities in every row of input column
    Args: 
     colData (array_like, 1D):Pandas Series of Data or Dataframe Column of text Data for function to be applied on
    Returns:
     result : list of dictionaries containing counts of extracted entities in every row of input column
    
    Example: 
    >>ft.extract_c(pd.Series(['hello my name is john', 'these are oranges', 'the sky is blue']))
    [{'PERSON': 1}, {}, {}]

    """
    result = []
    for d in entity_ls(colData):
        newd={}
        for k,v in d.items():
            newd[k]=len(v)
        result.append(newd)
    return result



def entity_ls(colData):
    """Entity List
    Returns list of all extracted entities in dictionaries for every row of input column
    Args: 
     colData (array_like, 1D):Pandas Series of Data or Dataframe Column of text Data for function to be applied on
    Returns:
     result : list of dictionaries containing list of extracted entities in every row of input column
    
    Example: 
    >>ft.extract_ls(pd.Series(['hello my name is john', 'these are oranges', 'the sky is blue']))
    [{'PERSON': ['john']}, {}, {}]

    """
    result = []
    for ls in extract_entity(colData):
        s=[]
        d={}
        for tup in ls:
            if tup[1] not in s:
                s.append(tup[1])
                d[tup[1]]=[tup[0]]
            else:
                d[tup[1]].append(tup[0])
        result.append(d)
    return result


def entity_count(colData, entity):
    """Specific Entity Count
    Returns list of sepcified extracted entities in every row of input column
    Args: 
     colData (array_like, 1D):Pandas Series of Data or Dataframe Column of text Data for function to be applied on
     
     entity: String name of entity
     
    Returns:
     result : list of dictionaries containing counts of extracted entity in every row of input column
    
    Example: 
    >>ft.entity_count(pd.Series(['hello my name is john', 'these are oranges', 'the sky is blue']),'PERSON')
    [1,0,0]

    """
    result=[]
    for i in entity_c(colData):
        count = 0
        if entity in i.keys():
            count+=i[entity]
        result.append(count)
    return result


def pos(colData):
    """Extracted Part of Speech
    Returns list of lists of extracted part of speeches in every row of input column
    Args: 
     colData (array_like, 1D):Pandas Series of Data or Dataframe Column of text Data for function to be applied on
     
     
    Returns:
     result : list of lists of extracted part of speeches in every row of input column
     
    Example: 
    >>ft.pos(pd.Series(['hello my name is john', 'these are oranges', 'the sky is blue']),'PERSON')
    [[('hello', 'NN'),
      ('my', 'PRP$'),
      ('name', 'NN'),
      ('is', 'VBZ'),
      ('john', 'JJ')],
     [('these', 'DT'), ('are', 'VBP'), ('oranges', 'NNS')],
     [('the', 'DT'), ('sky', 'NN'), ('is', 'VBZ'), ('blue', 'JJ')]]

    CC coordinating conjunction
    CD cardinal digit
    DT determiner
    EX existential there (like: “there is” … think of it like “there exists”)
    FW foreign word
    IN preposition/subordinating conjunction
    JJ adjective ‘big’
    JJR adjective, comparative ‘bigger’
    JJS adjective, superlative ‘biggest’
    LS list marker 1)
    MD modal could, will
    NN noun, singular ‘desk’
    NNS noun plural ‘desks’
    NNP proper noun, singular ‘Harrison’
    NNPS proper noun, plural ‘Americans’
    PDT predeterminer ‘all the kids’
    POS possessive ending parent’s
    PRP personal pronoun I, he, she
    PRP$ possessive pronoun my, his, hers
    RB adverb very, silently,
    RBR adverb, comparative better
    RBS adverb, superlative best
    RP particle give up
    TO, to go ‘to’ the store.
    UH interjection, errrrrrrrm
    VB verb, base form take
    VBD verb, past tense took
    VBG verb, gerund/present participle taking
    VBN verb, past participle taken
    VBP verb, sing. present, non-3d take
    VBZ verb, 3rd person sing. present takes
    WDT wh-determiner which
    WP wh-pronoun who, what
    WP$ possessive wh-pronoun whose
    WRB wh-abverb where, when"""
    result = []
    for str in colData:
        text = word_tokenize(str)
        pos = nltk.pos_tag(text)
        result.append(pos)
    return result


def pos_ls(colData):
    """List of Extracted Part of Speech
    Returns list of dictionaries of each part of speech containing all corresponding words, for every row of input column
    Args: 
     colData (array_like, 1D):Pandas Series of Data or Dataframe Column of text Data for function to be applied on
     
     
    Returns:
     result : list of dictionaries of each part of speeches with list of corresponding words in every row of input column
    Example: 
    >>ft.pos_ls(pd.Series(['hello my name is john', 'these are oranges', 'the sky is blue']),'PERSON')

    [{'NN': ['hello', 'name'], 'PRP$': ['my'], 'VBZ': ['is'], 'JJ': ['john']},
     {'DT': ['these'], 'VBP': ['are'], 'NNS': ['oranges']},
     {'DT': ['the'], 'NN': ['sky'], 'VBZ': ['is'], 'JJ': ['blue']}]
    """
    result=[]
    for ls in pos(colData):
        #st={"CC":0,'CD':0,'DT':0,'EX':0,'FW':0,'IN':0,'JJ':0,'JJR':0}
        s=[]
        d={}
        for tup in ls:
            if tup[1] not in s:
                s.append(tup[1])
                d[tup[1]]=[tup[0]]
           
            else:
                d[tup[1]].append(tup[0])
      
        result.append(d)
    return result




def pos_c(colData):
    """Part of Speech Count
    Returns list of dictionaries for every row of the input column, containing the number of words per extracted part of speech 
    Args: 
     colData (array_like, 1D):Pandas Series of Data or Dataframe Column of text Data for function to be applied on
    Returns:
     result : list of dictionaries containing counts of extracted pos in every row of input column
    
    Example: 
    >>ft.pos_c(pd.Series(['hello my name is john', 'these are oranges', 'the sky is blue']))
    [{'NN': 2, 'PRP$': 1, 'VBZ': 1, 'JJ': 1},
     {'DT': 1, 'VBP': 1, 'NNS': 1},
     {'DT': 1, 'NN': 1, 'VBZ': 1, 'JJ': 1}]

    """
    
    result = []
    for d in pos_ls(colData):
        newd={}
        for k,v in d.items():
            newd[k]=len(v)
        result.append(newd)
    return result
    
    
    
def pos_count(colData, pos):
    """Specific Part of Speech Count
    Returns list of sepcified part of speech in every row of input column
    Args: 
     colData (array_like, 1D):Pandas Series of Data or Dataframe Column of text Data for function to be applied on
     
     pos: String name of part of speech
     
    Returns:
     result : list of dictionaries containing counts of extracted part of speech in every row of input column
    
    Example: 
    >>ft.pos_count(pd.Series(['hello my name is john', 'these are oranges', 'the sky is blue']),'NN')
    [2, 0, 1]

    """
    result=[]
    for i in pos_c(colData):
        count = 0
        if pos in i.keys():
            count+=i[pos]
        result.append(count)
    return result


def kth_word(colData,k):
    """Kth Word
    Returns list of words at the Kth index of every row of the input column
    Args: 
     colData (array_like, 1D):Pandas Series of Data or Dataframe Column of text Data for function to be applied on
     
     k:  integer index
     
    Returns:
     result : list of dictionaries containing counts of extracted part of speech in every row of input column
    
    Example: 
    >>ft.kth_word(pd.Series(['hello my name is john', 'these are oranges', 'the sky is blue']),2)
    ['name', 'oranges', 'is']

    """
    result = []
    colData = tokenize_column(colData)
    for ls in colData:
        if k<len(ls):
            result.append(ls[k])
        else:
            result.append("")
                
    return result



    
def get_num_topics(corp, dic,dwb):
    """Helper Function for LDA - Calculate Ideal Number of Topics for LDA"""
    max=0
    topics=0
    print('Getting ideal number of topics...')
    for i in range(1,21):
        #building the topic model
        lda_model = gensim.models.ldamodel.LdaModel(corpus=corp, id2word=dic,num_topics=i,random_state=100,
                                                    update_every=1, chunksize=100,passes=10,alpha='auto',per_word_topics=True)

        #pprint(lda_model.print_topics())
        doc_lda = lda_model[corp]
        print(i)

        # Compute Perplexity
        print('\nPerplexity: ', lda_model.log_perplexity(corp))  # a measure of how good the model is. lower the better.

        # Compute Coherence Score
        coherence_model_lda = CoherenceModel(model=lda_model, texts= dwb, dictionary=dic, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        print('Coherence:',coherence_lda)
        if coherence_lda>max:
            max=coherence_lda
            
            topics=i
        #print('\nIdeal Number of Topics:',topics,'Coherence Score: ',coherence_lda)
    return topics







def make_bigrams(texts,bigram_mod):
    """Helper Function for LDA - Make Bigrams"""
    return [bigram_mod[doc] for doc in texts]


def make_trigrams(texts,bigram_mod):
    """Helper Function for LDA - Make Trigrams"""
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


def sentence_topic(ldamodel, corpus, texts):
    """Helper Function for LDA - Sentence Topic"""
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list            
        # print(row)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


    
def lda(colData): 
    """LDA
    Returns dataframe containing Dominant topics, keywords and most representative text for keywords using latent dirichlet allocation
    Args:
     colData (array_like, 1D):Pandas Series of text Data or Dataframe Column of text Data for function to be applied on
    
    Returns:
     df_dominant_topic : Pandas Dataframe with columns of text features
    
     
    """
    colData=normalize_corpus(colData)
    #Create Bigrams and Trigrams
    bigram = gensim.models.Phrases(colData, min_count=5,threshold = 100) #higher threshold fewer phrases
    trigram = gensim.models.Phrases(bigram[colData],threshold=100)

        #faster way to get sentence clubbed as trigtam/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod=gensim.models.phrases.Phraser(trigram)

    data_word_bigrams = make_bigrams(colData,bigram_mod)


        #Create dictionary
    id2word = corpora.Dictionary(data_word_bigrams)

        #Create corpus
    texts = data_word_bigrams

        #term doc frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    topics = get_num_topics(corpus, id2word, data_word_bigrams)    
    #print("corpus:",corpus[:1])
        #building the topic model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word,num_topics=10,random_state=100,
                                                update_every=1, chunksize=100,passes=10,alpha='auto',per_word_topics=True)

    pprint(lda_model.print_topics())
    #pp = lda_model.print_topics()
    doc_lda = lda_model[corpus]

        # Compute Perplexity
    print('\n Model Perplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

        # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts= data_word_bigrams, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\n Model Coherence Score: ', coherence_lda)
    
    df_topic_sents_keywords = sentence_topic(lda_model, corpus, colData)
    
    # Format
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

    #Most representative Sentence per topic
    # Display setting to show more characters in column
    pd.options.display.max_colwidth = 100

    output = pd.DataFrame()
    sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')


    for i, grp in sent_topics_outdf_grpd:
        output = pd.concat([output, grp.sort_values(['Perc_Contribution'], ascending=False).head(1)], 
                                                axis=0)

    # Reset Index    
    output.reset_index(drop=True, inplace=True)
    # Format
    output.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Representative Text"]
    
    #create most_rep_sentence column
    ls=[]
    for topic in df_dominant_topic['Dominant_Topic']:
        text=list(output['Representative Text'][output['Topic_Num']==topic])
        ls.append(text[0])
        #print("type: ", type(ls[0]))
    ls=pd.Series(ls)
    df_dominant_topic['Most Representative Text for Topic']=ls
   
   

# Format
    return df_dominant_topic
    





     



     
# Meta Attribute Table


def text_loader(df):
    """Text Loader
    Returns list of dataframes for every text column in the input dataframe, containing columns ofcompiled text features
    Args:
     dataframe : Pandas Dataframe that includes DateTime column(s) for function to be applied on
    
    Returns:
     result : list of dataframes for each DateTime column in the input dataset.
     
    Example: 
    *please refer to Date Time Demo Notebook for example of load_date_attributes output list of dataframes*

     
    """
    result=[]
    for col in df:
        print(col)
        
            
        if df[col].dtypes==str or df[col].dtypes== object:
            print("text col:", col)
            df[col]=df[col].fillna('')
            data = {col:df[col],
                    "Word 1":kth_word(df[col],0),
                    "Word 2":kth_word(df[col],1), "Word 3":kth_word(df[col],2), "Word 4":kth_word(df[col],3),"Word 5":kth_word(df[col],4),"Word 6":kth_word(df[col],5),
                    "Word 7":kth_word(df[col],6),"Word 7":kth_word(df[col],6),"Word 8":kth_word(df[col],7),"Word 9":kth_word(df[col],8),"Word 10":kth_word(df[col],9),
                    "Word 11":kth_word(df[col],10),"Word 12":kth_word(df[col],11),"Word 13":kth_word(df[col],12),"Word 14":kth_word(df[col],13),"Word 15":kth_word(df[col],14),
                    "Word 16":kth_word(df[col],15),"Word 17":kth_word(df[col],16),"Word 18":kth_word(df[col],17),"Word 19":kth_word(df[col],18),"Word 20":kth_word(df[col],19),
                    "Num Words":words_c(df[col]),
                    "Num Char":char_c(df[col]),
                    "Num stopwords":stopwords_c(df[col]),
                    "Avg words length":avg_length(df[col]),
                    "Num Digits":digits_c(df[col]),
                    "Num Alphabets":alph_c(df[col]),
                    "Num Spaces":space_c(df[col]),
                    "Num of Numbers": num_c(df[col]),
                    "Num Nouns":num_noun(df[col]),
                    "Entity: Person":entity_count(df[col], 'PERSON'),
                    "Entity: NORP":entity_count(df[col], 'NORP'),
                    "Entity: FAC":entity_count(df[col], 'FAC'),
                    "Entity: ORG":entity_count(df[col], 'ORG'),
                    "Entity: GPE":entity_count(df[col], 'GPE'),
                    "Entity: LOC ":entity_count(df[col], 'LOC'),
                    "Entity: PRODUCT":entity_count(df[col], 'PRODUCT'),
                    "Entity: EVENT":entity_count(df[col], 'EVENT'),
                    "Entity: WORK_OF_ART":entity_count(df[col], 'WORK_OF_ART'),
                    "Entity: LAW":entity_count(df[col], 'LAW'),
                    "Entity: LANGUAGE":entity_count(df[col], 'LANGUAGE'),
                    "Entity: DATE":entity_count(df[col], 'DATE'),
                    "Entity: TIME":entity_count(df[col], 'TIME'),
                    "Entity: PERCENT":entity_count(df[col], 'PERCENT'),
                    "Entity: MONEY":entity_count(df[col], 'MONEY'),
                    "Entity: CARDINAL":entity_count(df[col], 'CARDINAL'),
                    "Entity: ORDINAL":entity_count(df[col], 'ORDINAL'),
                    "POS: CC": pos_count(df[col], 'CC'),
                    "POS: CD": pos_count(df[col], 'CD'),
                    "POS: DT": pos_count(df[col], 'DT'),
                    "POS: EX": pos_count(df[col], 'EX'),
                    "POS: FW": pos_count(df[col], 'FW'),
                    "POS: IN": pos_count(df[col], 'IN'),
                    "POS: JJ": pos_count(df[col], 'JJ'),
                    "POS: JJR": pos_count(df[col], 'JJR'),
                    "POS: JJS": pos_count(df[col], 'JJS'),
                    "POS: LS": pos_count(df[col], 'LS'),
                    "POS: MD": pos_count(df[col], 'MD'),
                    "POS: NN": pos_count(df[col], 'NN'),
                    "POS: NNS": pos_count(df[col], 'NNS'),
                    "POS: NNP": pos_count(df[col], 'NNP'),
                    "POS: NNPS": pos_count(df[col], 'NNPS'),
                    "POS: PDT": pos_count(df[col], 'PDT'),
                    "POS: POS": pos_count(df[col], 'POS'),
                    "POS: PRP": pos_count(df[col], 'PRP'),
                    "POS: PRP$": pos_count(df[col], 'PRP$'),
                    "POS: RB": pos_count(df[col], 'RB'),
                    "POS: RBR": pos_count(df[col], 'RBR'),
                    "POS: RBS": pos_count(df[col], 'RBS'),
                    "POS: RP": pos_count(df[col], 'RP'),
                    "POS: TO": pos_count(df[col], 'TO'),
                    "POS: UH": pos_count(df[col], 'UH'),
                    "POS: VB": pos_count(df[col], 'VB'),
                    "POS: VBD": pos_count(df[col], 'VBD'),
                    "POS: VBG": pos_count(df[col], 'VBG'),
                    "POS: VBN": pos_count(df[col], 'VBN'),
                    "POS: VBP": pos_count(df[col], 'VBP'),
                    "POS: VBZ": pos_count(df[col], 'VBZ'),
                    "POS: WDT": pos_count(df[col], 'WDT'),
                    "POS: WP": pos_count(df[col], 'WP'),
                    "POS: WP$": pos_count(df[col], 'WP$'),
                    "POS: WRB": pos_count(df[col], 'WRB'),
                   }
        
            newdf = pd.DataFrame(data)
            totaldf = pd.concat([newdf,lda(df[col])],axis=1)
            result.append(totaldf)
            
    return result

