import streamlit as st
import pandas as pd
import numpy as np
import time
import ast
import random
from io import StringIO
from english_generator import EnglishAssignmentGenerator

random.seed(123)

st.header('English exercise generator')
'---'
# Add a checkbox to the sidebar:
st.sidebar.write("Choose the type of exercise:")

exercise_options = {
    'All': False,
    'select_word_verbs': False,
    'select_random_noun_phrase': False,
    'select_sentences': False,
    'write_missing_word': False
}

# Sidebar
selected_exercises = st.sidebar.checkbox("All", value=True)

if selected_exercises:
    selected_exercises = list(exercise_options.keys())[1:]  # Получить все ключи, кроме 'All'
else:
    for exercise in exercise_options:
        if exercise != 'All':
            exercise_options[exercise] = st.sidebar.checkbox(exercise)
            selected_exercises = [exercise for exercise, selected in exercise_options.items() if selected]

# Split exercises into pages
sentences_per_page  = st.sidebar.slider(
    'Select count sentences',
    5, 50, 10)


uploaded_file = st.file_uploader("select file in txt format:")
if uploaded_file is not None:
    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    # To read file as string:
    string_data = stringio.read()
else:
  st.warning('Please choose a file.')
  st.stop()

start_time = time.time()

# Function decorator for caching data
@st.cache_data
def transform(string_data):
    obj = EnglishAssignmentGenerator()
    df = obj.df_creation(string_data)
    return df


# Function for converting a string to a list
def convert_to_list(string_list):
    return ast.literal_eval(string_list)

# Function for shuffling characters in a string
def shuffle_string(text):
    words = text.split()
    shuffled_words = []
    for word in words:
        if len(word) > 5:
            shuffled_word = word[:2] + ''.join(random.sample(word[2:], len(word) - 2))
        else:
            shuffled_word = ''.join(random.sample(word, len(word)))
        
        shuffled_words.append(shuffled_word)   

    shuffled_text = ' '.join(shuffled_words)    
    return shuffled_text

df = transform(string_data)
end_time = time.time()
running_time = round(end_time - start_time, 1)

# Display the running time in Streamlit
st.write("Running time:", running_time, "seconds")

num_pages = len(df) // sentences_per_page
count_key = 0
task_total = 0
# Вывод каждой страницы
page = st.sidebar.selectbox('Выберите страницу', range(num_pages))

# Define exercise indices for the current page
start_index = page * sentences_per_page
end_index = (page + 1) * sentences_per_page

# Get exercises for the current page
tasks = df[start_index:end_index]

# Display exercises on the current page
st.subheader(f"Page {page}")

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
        'write_missing_word' if missing_word_answer != "Nan" else None
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
                answer = (st.selectbox('Choose the missing word:', ['–––'] + verb_tenses_options, 
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
                start_index = text.index(noun_chunks_selected)
                end_index = start_index + len(noun_chunks_selected)
                highlighted_sentence = f"{text[:start_index]}<mark>{noun_chunks_selected}</mark>{text[end_index:]}"
                st.markdown(highlighted_sentence, unsafe_allow_html=True)
                
            with col2:
                answer = (st.selectbox('Choose a part of speech', ['–––'] + noun_chunks_options, 
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
                "Which sentence is correct",
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
                show_text = st.checkbox("Show hint",
                                        key='show_text'+str(count_key))

                if show_text:
                    st.text("Make a word from letters: " + shuffle_string(missing_word_answer))

            with col2:    
                answer = st.text_input(
                    "Write the missing word",
                    key=count_key)

                if answer == missing_word_answer:
                    st.success('', icon="✅")
                    task_total += 1         
                else:
                    st.error('', icon="😟")

'---'
'Correct answers: ', task_total 
'Total Exercises:' , count_key        
'---'
if task_total == count_key:
    st.success('Success!')    

