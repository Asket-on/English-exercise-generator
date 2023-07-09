import numpy as np
import pandas as pd
import re
import os
import random
from time import time
import string
import spacy
import en_core_web_sm
from nltk.corpus import wordnet
import pyinflect

class MyClass:
    def method1(self):
        # малая модель spacy
        nlp = en_core_web_sm.load()

        file_path = r"C:\Users\m5612\YandexDisk\1_STUDIES\2_DS_YP_specialization\DS_YP_Project_masterskaya_NLP\Little_Red_Cap_ Jacob_and_Wilhelm_Grimm.txt"

        df = pd.DataFrame(columns=['raw'])

        with open(file_path, "r") as file:
            while line := file.readline():
                line = line.strip()
                if len(line)>0:
                    doc = nlp(line)
                    for sent in doc.sents:
                        df.loc[len(df), 'raw'] = sent.text

        def select_word_verbs(sentence):
            doc = nlp(sentence)
            verbs = [token for token in doc if token.pos_ == 'VERB']
            
            if not verbs:
                # Если в предложении нет глаголов, возвращаем исходное предложение
                return pd.Series([sentence, [], ''])
            
            # Случайный выбор глагола из списка
            verb = random.choice(verbs)
            
            verb_tenses_answer = verb.text
            verb_tenses_options = [verb._.inflect('VBP'), verb._.inflect('VBZ'), verb._.inflect('VBG'), verb._.inflect('VBD')]    
            verb_tenses_sent = sentence.replace(verb.text, '_____')

            return pd.Series([verb_tenses_sent, verb_tenses_options, verb_tenses_answer])


        def select_random_noun_phrase(sent):
            doc = nlp(sent)
            
            noun_chunks = [chunk for chunk in doc.noun_chunks]
            
            if len(noun_chunks) < 3:
                return pd.Series([None, [], None])
            
            selected_noun = random.choice(noun_chunks)
            
            noun_chunks_options = list(set(chunk.root.dep_ for chunk in noun_chunks))
            noun_chunks_selected = selected_noun.text
            noun_chunks_answer = selected_noun.root.dep_
            return pd.Series([noun_chunks_selected, noun_chunks_options, noun_chunks_answer])  


        def is_eligible_sentence(sentence):
            doc = nlp(sentence)
            main_words = [token for token in doc if token.pos_ in ['NOUN', 'VERB']]
            return len(main_words) > 2

        def generate_sentences(sentence):
            if not is_eligible_sentence(sentence):
                return []

            generated_sentences = []
            doc = nlp(sentence)
            main_words = [token for token in doc if token.pos_ in ['NOUN', 'VERB']]

            for word in main_words:
                antonyms = []
                for syn in wordnet.synsets(word.text):
                    for lemma in syn.lemmas():
                        if lemma.antonyms():
                            antonyms.append(lemma.antonyms()[0].name())
                if antonyms:
                    random.shuffle(antonyms)
                    num_replacements = random.randint(1, min(3, len(antonyms)))
                    replacements = antonyms[:num_replacements]
                    generated_sentence = sentence
                    for replacement in replacements:
                        generated_sentence = generated_sentence.replace(word.text, replacement, 1)
                    generated_sentences.append(generated_sentence)
            if len(generated_sentences) > 2:
                generated_sentences = random.sample(generated_sentences, 2)
            result = [sentence] + generated_sentences
            random.shuffle(result)
            return result


        def mising_word(sentence):
            doc = nlp(sentence)
            words = [token.text for token in doc]
            
            verb_indices = [i for i, token in enumerate(doc) if token.pos_ == 'VERB']
            noun_indices = [i for i, token in enumerate(doc) if token.pos_ == 'NOUN']
            adv_indices = [i for i, token in enumerate(doc) if token.pos_ == 'ADV']
            
            indices = verb_indices + noun_indices + adv_indices
            
            if len(indices) == 0:
                return pd.Series([0, 0])
            
            random_index = random.choice(indices)
            missing_word_answer = words[random_index]
            
            words[random_index] = '_____'
            missing_word_sentence = ' '.join(words)
            
            return pd.Series([missing_word_sentence, missing_word_answer])


        result_select_word_verbs = df['raw'].apply(select_word_verbs)
        result_noun_chunks = df['raw'].apply(select_random_noun_phrase)
        result_generate_sent = df['raw'].apply(generate_sentences)
        result_mising_word = df['raw'].apply(mising_word)

        df['verb_tenses_sent'] = result_select_word_verbs[0]
        df['verb_tenses_options'] = result_select_word_verbs[1]
        df['verb_tenses_answer'] = result_select_word_verbs[2]
        df['noun_chunks_selected'] = result_noun_chunks[0]
        df['noun_chunks_options'] = result_noun_chunks[1]
        df['noun_chunks_answer'] = result_noun_chunks[2]
        df['generate_sentences'] = result_generate_sent
        df['missing_word_sentence'] = result_mising_word[0]
        df['missing_word_answer'] = result_mising_word[1]

        df.to_csv('df_english.csv', index=False)