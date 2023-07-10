import numpy as np
import pandas as pd
import random
import en_core_web_sm
from pyinflect import getAllInflections
from sentence_splitter import SentenceSplitter, split_text_into_sentences
import nltk
from nltk.corpus import wordnet

nltk.download('wordnet')
nlp = en_core_web_sm.load()

class EnglishAssignmentGenerator:
    def df_creation(self, file_contents):
        # Split the text into sentences
        splitter = SentenceSplitter(language='en')
        sentences = splitter.split(text=file_contents)
        sentences = list(filter(lambda x: x != '', sentences))
        
        # Create a DataFrame to store the sentences
        df = pd.DataFrame({'raw': sentences})

        def select_word_verbs(sentence):
            # Select a random verb from the sentence
            doc = nlp(sentence)
            verbs = [token for token in doc if token.pos_ == 'VERB']
            
            if not verbs:
                # If there are no verbs in the sentence, return the original sentence
                return pd.Series([sentence, [], ''])
            
            verb = random.choice(verbs)
            
            # Get the verb's tense options and create a sentence with a blank for the verb
            verb_tenses_answer = verb.text
            verb_tenses_options = [inflection for inflection in getAllInflections(verb.text, 'V')]
            verb_tenses_sent = sentence.replace(verb.text, '_____')

            return pd.Series([verb_tenses_sent, verb_tenses_options, verb_tenses_answer])

        def select_random_noun_phrase(sent):
            # Select a random noun phrase from the sentence
            doc = nlp(sent)
            noun_chunks = [chunk for chunk in doc.noun_chunks]
            
            if len(noun_chunks) < 3:
                # If there are less than 3 noun chunks, return None
                return pd.Series([None, [], None])
            
            selected_noun = random.choice(noun_chunks)
            
            noun_chunks_options = list(set(chunk.root.dep_ for chunk in noun_chunks))
            noun_chunks_selected = selected_noun.text
            noun_chunks_answer = selected_noun.root.dep_
            return pd.Series([noun_chunks_selected, noun_chunks_options, noun_chunks_answer])  

        def is_eligible_sentence(sentence):
            # Check if a sentence is eligible for generating new sentences
            doc = nlp(sentence)
            main_words = [token for token in doc if token.pos_ in ['NOUN', 'VERB']]
            return len(main_words) > 2

        def generate_sentences(sentence):
            # Generate new sentences by replacing words with their antonyms
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
            # Replace a random word in the sentence with a blank
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

        # Apply the functions to create additional columns in the DataFrame
        result_select_word_verbs = df['raw'].apply(select_word_verbs)
        result_noun_chunks = df['raw'].apply(select_random_noun_phrase)
        result_generate_sent = df['raw'].apply(generate_sentences)
        result_mising_word = df['raw'].apply(mising_word)

        # Assign the results to new columns in the DataFrame
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

        return df
