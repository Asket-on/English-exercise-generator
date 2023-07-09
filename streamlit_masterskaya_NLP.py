import streamlit as st
import pandas as pd
import numpy as np
import ast
import random
from myclass import MyClass

#не разобрался еще как случайные выпадения в стримлите корректно делать, 
#в каждом новом цикле сбрасываются предыдущие задания. Поэтому зафиксировал рандом
random.seed(123)

# Использование класса
obj = MyClass()
obj.method1()


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
txt = st.text_area('Вставьте текст для создания упражнения')

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

df = pd.read_csv('df_english.csv')

df['verb_tenses_options'] = df['verb_tenses_options'].apply(convert_to_list)
df['noun_chunks_options'] = df['noun_chunks_options'].apply(convert_to_list)
df['generate_sentences'] = df['generate_sentences'].apply(convert_to_list)

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
        'write_missing_word' if missing_word_answer != '0' else None
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