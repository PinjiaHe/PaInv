"""Filtering sentences by their semantic and syntactic information"""
import requests
import urllib.request, urllib.error, urllib.parse
from bs4 import BeautifulSoup
import re
import json
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
from apted import APTED
from apted.helpers import Tree
from nltk.parse import CoreNLPParser
from nltk.stem import WordNetLemmatizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import wordnet
import nltk
from nltk.corpus import stopwords

def get_wordnet_pos(word):
    """Get pos tags of words in a sentence"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def treeToTree(tree):
    """Compute the distance between two trees by tree edit distance"""
    tree = tree.__str__()
    tree = re.sub(r'[\s]+',' ', tree)
    tree = re.sub('\([^ ]+ ', '(', tree)
    tree = tree.replace('(', '{').replace(')', '}')
    return next(map(Tree.from_text, (tree,)))

def treeDistance(tree1, tree2):
    """Compute distance between two trees"""
    tree1, tree2 = treeToTree(tree1), treeToTree(tree2)
    ap = APTED(tree1, tree2)
    return ap.compute_edit_distance()

def filtering_via_syntactic_and_semantic_information(pert_sent, synonyms):
    """Filter sentences by synonyms and constituency structure.
    Returns a dictionary of original sentence to list of filtered sentences
    """
    stopWords = list(set(stopwords.words('english')))
    syn_dic = {}
    filtered_sent = {}
    stemmer = SnowballStemmer("english")
    lemmatizer = WordNetLemmatizer()

    tokenizer = TreebankWordTokenizer()
    detokenizer = TreebankWordDetokenizer()

    # Run CoreNLPPArser on local host
    eng_parser = CoreNLPParser('http://localhost:9000')

    for original_sentence in list(pert_sent.keys()):
        # Create a dictionary from original sentence to list of filtered sentences
        filtered_sent[original_sentence] = []
        tokens_or = tokenizer.tokenize(original_sentence)
        # Constituency tree of source sentence
        source_tree = [i for i, in eng_parser.raw_parse_sents([original_sentence])]
        # Get lemma of each word of source sentence
        source_lem = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(original_sentence)]
        new_sents = pert_sent[original_sentence]
        target_trees_GT = []
        num = 50
        # Generate constituency tree of each generated sentence
        for x in range(int(len(new_sents)/num)):
            target_trees_GT[(x*num):(x*num)+num] = [i for i, in eng_parser.raw_parse_sents(new_sents[(x*num):(x*num)+num])]
        x = int(len(new_sents)/num)
        target_trees_GT[(x*num):] = [i for i, in eng_parser.raw_parse_sents(new_sents[(x*num):])]
        for x in range(len(new_sents)):
            s = new_sents[x]
            target_lem = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(s)]
            # If sentence is same as original sentence then filter that
            if s.lower()==original_sentence.lower():
                continue
            # If there constituency structure is not same, then filter
            if treeDistance(target_trees_GT[x],source_tree[0]) > 1:
                continue
            # If original sentence and generate sentence have same lemma, then filter
            if target_lem == source_lem:
                continue
            # Tokens of generated sentence
            tokens_tar = tokenizer.tokenize(s)
            for i in range(len(tokens_or)):
                if tokens_or[i]!=tokens_tar[i]:
                    word1 = tokens_or[i]
                    word2 = tokens_tar[i]
                    word1_stem = stemmer.stem(word1)
                    word2_stem = stemmer.stem(word2)
                    word1_base = WordNetLemmatizer().lemmatize(word1,'v')
                    word2_base = WordNetLemmatizer().lemmatize(word2,'v')
                    # If original word and predicted word have same stem, then filter
                    if word1_stem==word2_stem:
                        continue
                    # If they are synonyms of each other, the filter
                    syn1 = synonyms(word1_base)
                    syn2 = synonyms(word2_base)
                    if (word1 in syn2) or (word1_base in syn2) or (word2 in syn1) or (word2_base in syn1):
                        continue
                    if ((word1 in stopWords) or (word2 in stopWords) or (word1_stem in stopWords)
                        or (word2_stem in stopWords) or (word1_base in stopWords) or (word2_base in stopWords)):
                        continue
                    filtered_sent[original_sentence].append(s)
    return filtered_sent
