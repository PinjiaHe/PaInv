"""Generates syntactically similar sentences for a source sentence by removing words and phrases"""
from nltk.parse import CoreNLPParser
import nltk
import re
from apted import APTED
from apted.helpers import Tree
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
from nltk.corpus import stopwords

def replacenth(string, sub, wanted, n):
    """Replace nth word in a sentence
    
    string: Complete string
    sub: Substring to be replaced
    wanted: Replace by wanted
    n: index of the occurence of sub to be replaced
    """
    where = [m.start() for m in re.finditer(sub, string)][n-1]
    before = string[:where]
    after = string[where:]
    after = after.replace(sub, wanted, 1)
    newString = before + after
    return newString


def get_all(tree, detokenizer, stopWords):
    """Return all the phrases in the tree"""
    s = set()
    if str(tree)==tree:
        return s
    l = tree.label()
    if not (detokenizer.detokenize(tree.leaves()).lower() in stopWords):
        if (l=="JJR" or l=="JJS" or l=="JJ" or l=="NN" or l=="NNS" or l=="NNP" or l=="NNPS"
            or l=="RB" or l=="RBR" or l=="RBS" or l=="VB" or l=="VBD" or l=="VBG" or l=="VBN"
            or l=="VBP" or l=="VBZ" or l=="NP" or l=="VP" or l=="PP" or l=="ADVP"):
            s.add(detokenizer.detokenize(tree.leaves()))    
    for node in tree:
        s = s | get_all(node)
    return s

def generate_syntactically_similar_sentences(dataset):
    """Generate syntactically similar sentences for each sentence in the dataset.

    Returns dictionary of original sentence to list of generated sentences
    """
    # Run CoreNLPPArser on local host
    eng_parser = CoreNLPParser('http://localhost:9000')
    
    # Use nltk treebank tokenizer and detokenizer
    tokenizer = TreebankWordTokenizer()
    detokenizer = TreebankWordDetokenizer()

    # Stopwords from nltk
    stopWords = list(set(stopwords.words('english')))

    # Load dataset
    file = open(dataset,"r")

    dic = {}

    for line in file:
        sent = line.split("\n")[0]
        source_tree = eng_parser.raw_parse(sent)
        dic[line] = []
        for x in source_tree:
            phrases = get_all(x, detokenizer, stopWords)
            for t in phrases:
                if t=="'s":
                    continue
                for y in range(20):
                    try:
                        new_sent = replacenth(sent,t,"",y+1).replace("  "," ")
                        dic[line].append(sent)
                    except:
                        break
    return dic
