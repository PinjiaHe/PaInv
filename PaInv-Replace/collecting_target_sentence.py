import os, requests, uuid, json
from google.cloud import translate_v2 as translate

def BingTranslate(api_key, filtered_sent, language_from, language_to):
    """Bing Microsoft translator

    If you encounter any issues with the base_url or path, make sure
    that you are using the latest endpoint:
    https://docs.microsoft.com/azure/cognitive-services/translator/reference/v3-0-translate
    
    Arguments:
    api_key = Bing Microsoft Translate API key
    filtered_sent = dictionary of original sentence to list of filtered sentences
    language_from = Source language code
    language_to = Target language code

    returns translation dictionary from source sentence to target sentence
    """
    base_url = 'https://api.cognitive.microsofttranslator.com'
    path = '/translate?api-version=3.0'
    params = '&language='+ language_from +'&to=' + language_to
    constructed_url = base_url + path + params

    headers = {
        'Ocp-Apim-Subscription-Key': api_key,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }
    
    text = []

    for or_sent, filtered_sents in enumerate(filtered_sent):
        text.append(or_sent)
        text.extend(filtered_sents)

    body = [{'text': x} for x in text]
    # You can pass more than one object in body.
    
    request = requests.post(constructed_url, headers=headers, json=body)
    response = request.json()

    translation_dic = {}
    for i in range(len(text)):
        translation_dic[text[i]] = response[i]["translations"][0]["text"].replace('&#39;',"'").replace('&quot;',"'")
    return translation_dic

def GoogleTranslate(filtered_sent, source_language, target_language):
    """Google Translate, visit https://cloud.google.com/translate/docs to know pre-requisites
    
    Arguments:
    filtered_sent = dictionary of original sentence to list of filtered sentences
    source_language = Source language code
    target_language = Target language code

    returns translation dictionary from source sentence to target sentence
    """
    translate_client = translate.Client()
    translation_dic = {}
    
    for sent in filtered_sent.keys():
        sent1 = sent.split("\n")[0]
        ref_translation = ""
        ref_translation = translate_client.translate(sent1,target_language=target_language,
                                                     source_language=source_language)['translatedText'].replace('&#39;', "'").replace('&quot;', "'")
        translation_dic[sent] = ref_translation
        for new_s in filtered_sent[sent]:
            new_ref_translation = translate_client.translate(new_s,target_language=target_language,
                                                             source_language=source_language)['translatedText'].replace('&#39;',"'").replace('&quot;',"'")
            translation_dic[new_s] = new_ref_translation
    return translation_dic    

def collect_target_sentences(translator, filtered_sent, source_language, target_language, api_key=None):
    """Return Translation dic for a translator"""
    if translator == 'Google':
        return GoogleTranslate(filtered_sent, source_language, target_language)
    if translator == 'Bing':
        return BingTranslate(api_key, filtered_sent, source_language, target_language)
