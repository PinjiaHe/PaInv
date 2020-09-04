"""Filtering sentences by their semantic and syntactic information"""
def filtering_via_syntactic_and_semantic_information(syntactically_similar_sentences, threshold):
    """Filtering if the difference between length of original and generated sentence is < threshold
    Returns a dictionary of original sentence to list of filtered sentences
    """
    filtered_sent = {}
    for sent in syntactically_similar_sentences.keys():
        filtered_sent[sent] = []
        for similar_sent in syntactically_similar_sentences[sent]:
            if len(sent) - len(similar_sent) > threshold:
                filtered_sent[sent].append(similar_sent)
    return filtered_sent
