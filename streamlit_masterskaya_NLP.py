import streamlit as st
import pandas as pd
import numpy as np
import time
import ast
import random
from io import StringIO
#from myclass import EnglishAssignmentGenerator





import en_core_web_sm
from pyinflect import getAllInflections
from sentence_splitter import SentenceSplitter, split_text_into_sentences
import nltk
from nltk.corpus import wordnet

nltk.download('wordnet')
nlp = en_core_web_sm.load()

def df_creation(file_contents):
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





#не разобрался еще как случайные выпадения заданий в стримлите корректно делать, 
#в каждом новом цикле сбрасываются предыдущие задания. Поэтому зафиксировал рандом
random.seed(123)

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    #st.write(stringio)
    # To read file as string:
    string_data = stringio.read()
    #st.write('string_data', string_data)


start_time = time.time()
@st.cache_data
def transform(string_data):
    #obj = EnglishAssignmentGenerator()
    df = df_creation(string_data)
    return df


# Функция для преобразования строки в список
def convert_to_list(string_list):
    return ast.literal_eval(string_list)

#функция, которая перемешивает буквы в строке
def shuffle_string(text):
    # Преобразуем строку в список слов
    words = text.split()
    
    # Перемешиваем каждое слово в списке
    shuffled_words = []
    for word in words:
        if len(word) > 5:
            # Если слово больше 5 символов, оставляем первые две буквы на месте
            shuffled_word = word[:2] + ''.join(random.sample(word[2:], len(word) - 2))
        else:
            # Если слово не больше 5 символов, перемешиваем все его буквы
            shuffled_word = ''.join(random.sample(word, len(word)))
        
        shuffled_words.append(shuffled_word)   
    # Соединяем перемешанные слова обратно в строку
    shuffled_text = ' '.join(shuffled_words)    
    return shuffled_text

st.header('Генератор упражнений по английскому')
#in developing
#txt = st.text_area('Вставьте текст для создания упражнения')

'---'
# Add a checkbox to the sidebar:
st.sidebar.write("Выберите тип упражнения")

exercise_options = {
    'All': False,
    'select_word_verbs': False,
    'select_random_noun_phrase': False,
    'select_sentences': False,
    'write_missing_word': False
}

#sidebar
selected_exercises = st.sidebar.checkbox("All", value=True)

if selected_exercises:
    selected_exercises = list(exercise_options.keys())[1:]  # Получить все ключи, кроме 'All'
else:
    for exercise in exercise_options:
        if exercise != 'All':
            exercise_options[exercise] = st.sidebar.checkbox(exercise)
            selected_exercises = [exercise for exercise, selected in exercise_options.items() if selected]

# Разделение упражнений на страницы
sentences_per_page  = st.sidebar.slider(
    'Select count sentences',
    5, 50, 10)

df = transform(string_data)
#st.write(df.info())
#st.write(type(df.iloc[2, 1]))
#df['verb_tenses_options'] = df['verb_tenses_options'].apply(convert_to_list)
#df['noun_chunks_options'] = df['noun_chunks_options'].apply(convert_to_list)
#df['generate_sentences'] = df['generate_sentences'].apply(convert_to_list)
end_time = time.time()
running_time = round(end_time - start_time, 1)

# Display the running time in Streamlit
st.write("Running time:", running_time, "seconds")

num_pages = len(df) // sentences_per_page
count_key = 0
task_total = 0
# Вывод каждой страницы
page = st.sidebar.selectbox('Выберите страницу', range(num_pages))

# Определение индексов упражнений для текущей страницы
start_index = page * sentences_per_page
end_index = (page + 1) * sentences_per_page

# Получение упражнений для текущей страницы
tasks = df[start_index:end_index]

# Вывод упражнений на текущей странице
st.subheader(f"Page {page}")
df.head()
for _, task in tasks.iterrows():    
    
    verb_tenses_options = task['verb_tenses_options']
    verb_tenses_answer = task['verb_tenses_answer']
    noun_chunks_selected = task['noun_chunks_selected']
    noun_chunks_options = task['noun_chunks_options']
    noun_chunks_answer = task['noun_chunks_answer']
    generate_sentences = task['generate_sentences']
    missing_word_sentence = task['missing_word_sentence']
    missing_word_answer = task['missing_word_answer']


    available_exercises = [
        'select_word_verbs' if len(verb_tenses_options) > 0 else None,
        'select_random_noun_phrase' if len(noun_chunks_options) > 0 else None,
        'select_sentences' if len(generate_sentences) > 0 else None,
        'write_missing_word' if missing_word_answer != "0" else None
    ]

    if selected_exercises[0] == 'All':
        common_exercises = available_exercises
    else:  
        common_exercises = list(set(available_exercises) & set(selected_exercises))
    if len(common_exercises) == 0:
        task['raw']
    else:
        random_exercise = random.choice(common_exercises)
        if random_exercise == 'select_word_verbs':
            count_key += 1
            '---'
            col1, col2 = st.columns([7, 3])
            with col1:
                st.write('')
                st.write(task['verb_tenses_sent'])
                
            with col2:
                answer = (st.selectbox('Выберите пропущенное слово:', ['–––'] + verb_tenses_options, 
                                                    key=count_key))
                if answer == '–––':
                    pass
                elif answer == verb_tenses_answer:
                    st.success('', icon="✅")
                    task_total += 1
                else:
                    st.error('', icon="😟")

        elif random_exercise == 'select_random_noun_phrase':
            count_key += 1
            '---'
            col1, col2 = st.columns([7, 3])
            text = task['raw']
            with col1:
                st.write('')
                # Определяем позицию выделенной части
                start_index = text.index(noun_chunks_selected)
                end_index = start_index + len(noun_chunks_selected)
                # Формируем строку с выделением
                highlighted_sentence = f"{text[:start_index]}<mark>{noun_chunks_selected}</mark>{text[end_index:]}"
                st.markdown(highlighted_sentence, unsafe_allow_html=True)
                
            with col2:
                answer = (st.selectbox('Выберите часть речи', ['–––'] + noun_chunks_options, 
                                                    key=count_key))
                if answer == '–––':
                    pass
                elif answer == noun_chunks_answer:
                    st.success('', icon="✅")
                    task_total += 1
                else:
                    st.error('', icon="😟")
        
        elif random_exercise == 'select_sentences':
            count_key += 1
            '---'
            answer = st.radio(
                "Какое предложение верно",
                ['–––'] + generate_sentences, key=count_key)
            if answer == '–––':
                pass            
            elif answer == task['raw']:
                st.success('', icon="✅")
                task_total += 1
            else:
                st.error('', icon="😟")

        elif random_exercise == 'write_missing_word':
            count_key += 1
            '---'
            col1, col2 = st.columns([7, 3])
            with col1:                 
                missing_word_sentence
                show_text = st.checkbox("Показать подсказку",
                                        key='show_text'+str(count_key))

                if show_text:
                    st.text("Составьте слово из букв: " + shuffle_string(missing_word_answer))

            with col2:    
                answer = st.text_input(
                    "Напишите пропущенное слово",
                    key=count_key)

                if answer == missing_word_answer:
                    st.success('', icon="✅")
                    task_total += 1         
                else:
                    st.error('', icon="😟")

'---'
'Правильных ответов: ', task_total 
'Всего упражнений:' , count_key        
'---'
if task_total == count_key:
    st.success('Успех!')    

st.button("Re-run")
