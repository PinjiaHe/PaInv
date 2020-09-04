with open("politics_hindi_translations.dat", 'wb') as f: #change politics or hindi suitablly
    pickle.dump(translation_dic, f)         

def detecting_translation_errors(filtered_sent, translation_dic, filename):
    """Detect translation errors by comparing translation of original sentence with
    generated sentence

    filtered_sent: dictionary of original sentence to list of filtered sentences
    translation_dic: Dictionary of sentence to its tranlation
    filename: File's name where suspected issues will be wrote
    """
    f = open(filename, "a")
    for sent in filtered_sent.keys():
        sent1 = sent.split("\n")[0]
        ref_translation = translation_dic[sent]
        for new_s in filtered_sent[sent]:
            new_ref_translation = translation_dic[new_s]
            if ref_translation==new_ref_translation:
                f.write(sent)
                f.write(new_s)
                f.write(ref_translation)
                f.write(" ")
    f.close()
