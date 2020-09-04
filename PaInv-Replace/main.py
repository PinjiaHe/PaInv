import pickle
from generate_syntactically_similar_sentences import generate_syntactically_similar_sentences
from filtering_via_syntactic_and_semantic_information import filtering_via_syntactic_and_semantic_information
from filter_by_sentence_embeddings import filter_by_sentence_embeddings
from collect_target_sentences import collect_target_sentences
from detecting_translation_errors import detecting_translation_errors

# Path of dataset to be used to find translation errors
dataset = "dataset/business"

num_of_perturb = 50 # Number of perturbation for each word in the sentence

syntactically_similar_sentences = generate_syntactically_similar_sentences(num_of_perturb, dataset)

# File where a dictionary of generated sentences corresponding to original sentence will be saved
with open("data/business_bert_output.dat", 'wb') as f:
	pickle.dump(syntactically_similar_sentences, f)

# Load dictionary of synonyms for each word
with open("data/synonyms.dat", 'rb') as f:
	synonyms = pickle.load(f)

# Filtering by synonyms and filtering by constituency structure
filtered_sentences = filtering_via_syntactic_and_semantic_information(syntactically_similar_sentences,
	                                                                  synonyms)

# File where a dictionary of filtered sentences corresponding to original sentence will be saved
with open("data/business_filter_output.dat", 'wb') as f:
	pickle.dump(filtered_sentences, f)

# optional step: filtering by sentence embeddings

threshold = 0.9    # Choose threshold for filtering

# Run install_USE.sh and install Universal sentence encoder before running the code below
filtered_sentences = filter_by_sentence_embeddings(filtered_sentences, threshold)

# File where a dictionary of filtered sentences corresponding to original sentence will be saved
with open("data/business_filter_output.dat", 'wb') as f:
	pickle.dump(filtered_sentences, f)

# Bing Translate
bing_translate_api_key = 'enter the API key for Bing Microsoft Translate'

# target language: Chinese
source_language = 'en'
target_language = 'zh-Hans'

target_sentences = collect_target_sentences("Bing", filtered_sentences, source_language,
	                                        target_language, bing_translate_api_key)

# Save translations locally for later use
with open("data/business_chinese.dat", 'wb') as f:
    pickle.dump(target_sentences, f)

# Google Translate

# target language: Hindi
source_language = 'en'
target_language = 'hi'

target_sentences = collect_target_sentences("Google", filtered_sentences, source_language,
	                                        target_language)

# Save translations locally for later use
with open("data/business_hindi.dat", 'wb') as f:
    pickle.dump(target_sentences, f)

filename = "business_hindi_errors"

# Write translation errors in filename
detecting_translation_errors(filtered_sentences, target_sentences, filename)
