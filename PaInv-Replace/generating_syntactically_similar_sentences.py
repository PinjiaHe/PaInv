"""Generates perturbed sentences for a source sentence using BERT"""
import nltk
import numpy as np
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import string
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
from bert_embedding import BertEmbedding
import nltk
from nltk.corpus import stopwords
from nltk.corpus import words

def perturb(sentence, bertmodel, num):
	"""Generate a list of similar sentences by BERT
	
	Arguments:
	sentence: Sentence which needs to be perturbed
	bertModel: MLM model being used (BERT here)
	num: Number of perturbations required for a word in a sentence
	"""

	# Tokenize the sentence
	tokens = tokenizer.tokenize(sent)
	pos_inf = nltk.tag.pos_tag(tokens)

	# the elements in the lists are tuples <index of token, pos tag of token>
	bert_masked_indexL = list()

	# collect the token index for substitution
	for idx, (word, tag) in enumerate(pos_inf):
		if (tag.startswith("JJ") or tag.startswith("JJR") or tag.startswith("JJS")
			or tag.startswith("PRP") or tag.startswith("PRP$") or  tag.startswith("RB")
			or tag.startswith("RBR") or tag.startswith("RBS") or tag.startswith("VB") or
			tag.startswith("VBD") or tag.startswith("VBG") or tag.startswith("VBN") or
			tag.startswith("VBP") or tag.startswith("VBZ") or tag.startswith("NN") or
			tag.startswith("NNS") or tag.startswith("NNP") or tag.startswith("NNPS")):

			tagFlag = tag[:2]

			if (idx!=0 and idx!=len(tokens)-1):
				bert_masked_indexL.append((idx, tagFlag))

	bert_new_sentences = list()

	# generate similar setences using Bert
	if bert_masked_indexL:
		bert_new_sentences = perturbBert(sent, bertmodel, num, bert_masked_indexL)
	return bert_new_sentences


def perturbBert(sent, model, num, masked_indexL):
	"""Generate a list of similar sentences by Bert
	
	Arguments:
	sent: sentence which need to be perturbed
	model: MLM model
	num: Number of perturbation for each word
	masked_indexL: List of indexes which needs to be perturbed
	"""

	global num_words_perturb
	new_sentences = list()
	tokens = tokenizer.tokenize(sent)

	# set of invalid characters
	invalidChars = set(string.punctuation)

	# for each idx, use Bert to generate k (i.e., num) candidate tokens
	for (masked_index, tagFlag) in masked_indexL:
		original_word = tokens[masked_index]
		# Getting the base form of the word to check for it's synonyms
		low_tokens = [x.lower() for x in tokens]		
		low_tokens[masked_index] = '[MASK]'
		# Eliminating cases for "'s" as Bert does not work well on these cases.		
		if original_word=="'s":
			continue
		# Eliminating cases of stopwords
		if original_word in stopWords:
			continue
		# try whether all the tokens are in the vocabulary
		try:
			indexed_tokens = berttokenizer.convert_tokens_to_ids(low_tokens)
			tokens_tensor = torch.tensor([indexed_tokens])
			prediction = model(tokens_tensor)

		except KeyError as error:
			print ('skip sent. token is %s' % error)
			continue
		
		# Get the similar words
		topk_Idx = torch.topk(prediction[0, masked_index], num)[1].tolist()
		topk_tokens = berttokenizer.convert_ids_to_tokens(topk_Idx)
		num_words_perturb += 1
		# Remove the tokens that only contains 0 or 1 char (e.g., i, a, s)
		topk_tokens = list(filter(lambda x:len(x)>1, topk_tokens))
		# Remove the cases where predicted words are synonyms of the original word or both words have same stem.
		# generate similar sentences
		for x in range(len(topk_tokens)):
			t = topk_tokens[x]
			if any(char in invalidChars for char in t):
				continue
			tokens[masked_index] = t
			new_pos_inf = nltk.tag.pos_tag(tokens)

			# only use the similar sentences whose similar token's tag is still JJ, JJR, JJS, PRP, PRP$, RB, RBR, RBS, VB, VBD, VBG, VBN, VBP, VBZ, NN, NNP, NNS or NNPS
			if (new_pos_inf[masked_index][1].startswith(tagFlag)):
				new_sentence = detokenizer.detokenize(tokens)
				new_sentences.append(new_sentence)
		tokens[masked_index] = original_word

	return new_sentences

def generate_syntactically_similar_sentences(num_of_perturb, dataset):
	"""Generate syntactically similar sentences for each sentence in the dataset.

	Returns dictionary of original sentence to list of generated sentences
	"""
	# Use nltk treebank tokenizer and detokenizer
	tokenizer = TreebankWordTokenizer()
	detokenizer = TreebankWordDetokenizer()

	# Stopwords from nltk
	stopWords = list(set(stopwords.words('english')))

	# File from which sentences are read
	file = open(dataset, "r")

	# when we use Bert
	berttokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
	bertmodel = BertForMaskedLM.from_pretrained('bert-large-uncased')
	bertmodel.eval()

	# Number of perturbations you want to make for a word in a sentence
	dic = {}
	num_of_perturb = 50
	num_sent = 0
	for line in file:
		s_list = line.split("\n")
		source_sent = s_list[0]
		# Generating new sentences using BERT
		new_sents = perturb(source_sent, bertmodel, num_of_perturb)
		dic[line] = new_sents		
		if new_sents != []:
			num_sent += 1
	return dic
