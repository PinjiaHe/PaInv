"""Return filtered sentences after filtering by sentence embeddings"""
from use.use import UseSimilarity

def filter_by_sentence_embeddings(sentences_dic, threshold):
	"""Filter sentence by ensuring similarity less than threshold
	Returns dictionary of original sentence to list og filtered sentences
	"""
	filtered_sentences = {}
	similarity_model = UseSimilarity('USE')
	for original_sent, generated_sents in enumerate(sentences_dic):
		filtered_sentences[original_sent] = []
		for sent in generated_sents:
			if similarity_model.get_sim([original_sent, sent]) < threshold:
				filtered_sentences[original_sent].append(sent)
	return filtered_sentences