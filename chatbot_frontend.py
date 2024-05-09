import streamlit as st 
import chatbot_backend as glib 
import time
import random
import re

# random.seed(46)

st.set_page_config(page_title="Vaccine Echo Chamber💉") 

def set_sidebar_width(width):
    st.markdown(
        f"""
        <style>
            .css-1d391kg {{
                width: {width}px;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )
set_sidebar_width(400)


assistants = [
    {"name": "Mr.Dinosaur", "avatar": '🦕'},
    {"name": "Ms.Bird", "avatar": '🐤'},
    {"name": "Mrs.Tiger", "avatar": '🐯'},
    {"name": "Dr.Panda", "avatar": '🐼'},
    {"name": "Dr.Robot", "avatar": '🤖'},
    {"name": "Dr.Unicorn", "avatar": '🦄'},
    {"name": "Mr.Elephant", "avatar": '🐘'},
    {"name": "Ms.Butterfly", "avatar": '🦋'},
    {"name": "Mrs.Dolphin", "avatar": '🐬'},
    {"name": "Dr.Owl", "avatar": '🦉'},
    {"name": "Mr.Kangaroo", "avatar": '🦘'},
    {"name": "Ms.Penguin", "avatar": '🐧'},
    {"name": "Mrs.Koala", "avatar": '🐨'},
    {"name": "Dr.Flamingo", "avatar": '🦩'},
    {"name": "Mr.Giraffe", "avatar": '🦒'},
    {"name": "Ms.Hedgehog", "avatar": '🦔'},
    {"name": "Mrs.Raccoon", "avatar": '🦝'},
    {"name": "Dr.Sloth", "avatar": '🦥'},
    {"name": "Mr.Hippo", "avatar": '🦛'},
    {"name": "Ms.Parrot", "avatar": '🦜'}
]


if 'gpt' not in st.session_state.keys():
    st.session_state.gpt = glib.get_gpt()

if 'phi2' not in st.session_state.keys():
    st.session_state.phi2 = glib.get_phi2()

if 'memory' not in st.session_state.keys():
    st.session_state.memory = glib.get_memory(st.session_state.gpt)

# chat_history: {"type":'message'/'warning', "name":assistant['name'], "text":comment, "avatar":assistant["avatar"], "quote":''}
if 'chat_history' not in st.session_state.keys(): 
    st.session_state.chat_history = []

if 'user_name' not in st.session_state.keys():
    st.session_state.user_name = None

def add_sign_input(input_text):
    if input_text and not input_text[-1] in '.!?':
        input_text += '.'
    return input_text

def get_last_sentence_part(remarker):
    # ori_string = st.session_state.chat_history[-1]['text'][:length]
    # parts = ori_string.split('.')
    # parts = [part for part in parts if part.strip()]
    # if len(parts) <= 1:
    #     return ori_string
    # selected_part = random.choice(parts[:-1]).strip() + '.'
    # return selected_part

    remarker_name = remarker['name']
    for message in st.session_state.chat_history[::-1]:
        if message['type'] == 'message':
            if remarker_name != message['name'] and len(message['text']) > 20:
                return message["text"]
        
def identity_changing(question):
    new_question = question.replace("i ", "I ")
    new_question = new_question.replace("I am ", "Someone is ")
    new_question = new_question.replace("i am ", "Someone is ")
    new_question = new_question.replace("I'm ", "Someone is ")
    new_question = new_question.replace("i'm ", "Someone is ")
    new_question = new_question.replace("Im ", "Someone is")
    new_question = new_question.replace("im ", "Someone is ")
    new_question = new_question.replace("my ", "Someone's ")
    new_question = new_question.replace("My ", "Someone's ")
    new_question = new_question.replace("I ", "Someone ")
    return new_question
        

def do_a_static_comment(comment, assistant):
    placeholder = st.empty()
    with placeholder.container():
        with st.chat_message(assistant["name"], avatar = assistant["avatar"]): 
            with st.spinner(assistant['name'] + " is typing..."):
                st.session_state.chat_history.append({"type":'message', "name":assistant['name'], "text":comment, "avatar":assistant["avatar"], "quote":None})
                time.sleep(1.5)
                st.markdown(comment)
                


def do_a_direct_response(question, responser):
    placeholder = st.empty()
    with placeholder.container():
        with st.chat_message(responser['name'], avatar = responser["avatar"]): 
            with st.spinner(responser['name'] + " is typing..."):
                print(question)
                new_question = identity_changing(question)
                print(new_question)
                chat_response = glib.get_bad_vaccine_response(input_text=new_question, phi2=st.session_state.phi2, memory=st.session_state.memory)
                sentences = re.split(r'[,.!?]+', chat_response)
                sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    placeholder.empty()

    for i in range(len(sentences)):
        sentence = sentences[i]
        placeholder = st.empty()
        with placeholder.container():
            with st.chat_message(responser["name"], avatar = responser["avatar"]): 
                with st.spinner(responser['name'] + " is typing..."): 

                    if i == 0: 
                        sentence = sentence.replace(new_question, "emmm...")
                        st.session_state.chat_history.append({"type":'message', "name": responser["name"], "text": sentence, "avatar": responser["avatar"], "quote":question})
                        st.markdown(f"\>\>\>''*{question}*''")
                    else:
                        st.session_state.chat_history.append({"type":'message', "name": responser["name"], "text": sentence, "avatar": responser["avatar"], "quote":None})
                        
                    time.sleep(random.uniform(1, 2))
                    placeholder = st.empty()
                    full_response = ''
                    for item in sentence.split():
                        full_response += item + ' '
                        placeholder.markdown(full_response)
                        time.sleep(random.uniform(0, 0.5))
                    placeholder.markdown(full_response)

def do_a_last_sentence_remark(remarker, times):
    placeholder = st.empty()
    with placeholder.container():
        with st.chat_message(remarker["name"], avatar = remarker["avatar"]): 
            with st.spinner(remarker['name'] + " is typing..."):
                last_sentence_part = get_last_sentence_part(remarker)
                chat_remark = glib.get_bad_vaccine_remark(message=last_sentence_part, times=times, phi2=st.session_state.phi2, memory=st.session_state.memory)
                sentences = re.split(r'[,.!?]+', chat_remark)
                sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    placeholder.empty()

    for i in range(len(sentences)):
        sentence = sentences[i]
        placeholder = st.empty()
        with placeholder.container():
            with st.chat_message("assistant", avatar = remarker["avatar"]): 
                with st.spinner(remarker['name'] + " is typing..."): 

                    if i == 0: 
                        if sentence.startswith(last_sentence_part):
                            sentence = sentence.replace(last_sentence_part, 'I agree.')
                        st.session_state.chat_history.append({"type":'message', "name": remarker["name"], "text": sentence, "avatar": remarker["avatar"], "quote":last_sentence_part})
                        st.markdown(f"\>\>\>''*{last_sentence_part}*''")
                    else:
                        st.session_state.chat_history.append({"type":'message', "name": remarker["name"], "text": sentence, "avatar": remarker["avatar"], "quote":None})

                    time.sleep(random.uniform(1, 2))
                    placeholder = st.empty()
                    full_remark = ''
                    for item in sentence.split():
                        full_remark += item + ' '
                        placeholder.markdown(full_remark)
                        time.sleep(random.uniform(0, 0.5))
                    placeholder.markdown(full_remark)
    

def do_a_funny_remark(assistant):
    placeholder = st.empty()
    with placeholder.container():
        with st.chat_message(assistant['name'], avatar = assistant["avatar"]): 
            with st.spinner(assistant['name'] + " is typing..."):
                chat_remark = glib.get_funny_remark()
                st.session_state.chat_history.append({"type":'message', "name": assistant['name'], "text": chat_remark, "avatar": assistant["avatar"], "quote":None})
                time.sleep(2)
                placeholder = st.empty()
                full_remark = ''
                for item in chat_remark.split():
                    full_remark += item + ' '
                    placeholder.markdown(full_remark)
                    time.sleep(random.uniform(0, 0.1))
                placeholder.markdown(full_remark)

def do_a_dummy_agree():
    if random.randint(0,1) < 0.4: return
    print('>>>>>>>>>>DUMMY<<<<<<<<<<<')
    agree_template = [
        f'yes you are right.',
        f'really?',
        f'Yeah',
        f'Yes',
        f'Oh, wtf?',
        f'yes I agree',
        f'Agree',
        f'Oh My God',
        f'Oh My Gosh',
        f'That\'s right',
        f'serious??',
        f'Ohhhhh',
        f'wow really?',
        f'that\'s correct',
        f'fair enough',
        f'that makes sense',
        f'oh god',
        f'what??'
    ]
    assistant = random.choice(assistants)
    hidden_assistant = random.choice(assistants)
    index = random.randint(0, len(agree_template) - 1)
    hidden_index = random.randint(0, len(agree_template) - 1)
    placeholder = st.empty()
    with placeholder.container():
        with st.chat_message(assistant["name"], avatar = assistant["avatar"]): 
            with st.spinner(assistant['name'] + " is typing..."):
                content = get_last_sentence_part(assistant)
                st.session_state.chat_history.append({"type":'message', "name":assistant['name'], "text":agree_template[index], "avatar":assistant["avatar"], "quote":content})
                st.session_state.chat_history.append({"type":'message', "name":hidden_assistant['name'], "text":agree_template[hidden_index], "avatar":hidden_assistant["avatar"], "quote":content})
                time.sleep(1.5)
                st.markdown(f"\>\>\>''*{content}*''")
                st.markdown(agree_template[index])
                

with st.sidebar:
    # st.title('Welcome to the Vaccine Echo Chamber 💉💉💉')
    st.sidebar.markdown("<h1 style='margin-top: -50px;'>Welcome to the Vaccine Echo Chamber 💉💉💉</h1>", unsafe_allow_html=True)
    st.success('Successfully Connected! Talk anything related to the vaccines!', icon='✅')
    st.warning('Only for research purpose! Do not trust anyone!!!', icon='❌')

    st.subheader('Chat Examples')
    example_questions = [
        "Tell me your attitude about vaccines.",
        "Describe how you feel about the effectiveness of vaccines.",
        "Explain any hesitations you have about getting certain vaccines.",
        "Tell me about your experience with receiving your vaccine.",
		"Share any stories or experiences you've had with vaccinations.",
    ]
    for question in example_questions:
        st.markdown(f"- {question}")

    people = st.sidebar.slider('Max People', min_value=5, max_value=10, value=7, step=1)

# add a backgroung image
st.image("background.png", use_column_width=True)


st.warning('Please end your input with a punctuation mark (., !, or ?).')


# print all of the historical conversations
for message in st.session_state.chat_history: 
    if message["type"] == 'message':
        with st.chat_message(message["name"], avatar = message["avatar"]): 
            if message["quote"] != None:
                quote = message["quote"]
                st.markdown(f"\>\>\>''*{quote}*''")
            st.markdown(message["text"]) 
    elif message["type"] == 'warning':
         st.success('Message Send Successfully! Do not trust anyone in the room!', icon='✅')
    else:
        continue



# clear all of hte chat history
def clear_chat_history():
    st.session_state.chat_history = [] 
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# saving all of the chat history
def save_chat_history():
    chat_string = ""
    for message in st.session_state.chat_history:
        if message['type'] == 'message':
            chat_string += f"{message['name']}: {message['text']}\n"
    time_stamp = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(time.time()))
    file_path = f'.../echo_chamber/experiment/results/{st.session_state.user_name}_{time_stamp}.txt'
    with open(file_path, 'w') as file:
        file.write(chat_string)
user_name = st.sidebar.text_input('⬇️ Please input your nickname here')
if user_name:
    st.session_state.user_name = user_name
st.sidebar.button('Save Chat History', on_click=save_chat_history)     

# initialize input box
input_text = st.chat_input("Chat with others here") 

###########################################################################################################

# auto: 1 question, 1 answer, 2 remarks;
# general: 1 question, 1 answer, 2-3 remarks, 1 answer(or funny), 2-3 remarks;

# beginning Auto conversation
if 'start' not in st.session_state.keys(): 
# if 'start' not in st.session_state.keys() and st.session_state.user_name != None: 
    st.session_state.start = True
    # time.sleep(3.5)
    asker = random.choice(assistants)
    responser = random.choice(assistants)
    while(asker == responser): responser = random.choice(assistants)
    static_question = 'Hi, tell me your attitude toward vaccines.'

    do_a_static_comment(static_question, asker)

    do_a_direct_response(static_question, responser)
    do_a_dummy_agree()

    # generate remarks randomly
    times = random.randint(2, 2)
    for i in range(times):
        assistant = random.choice(assistants)
        do_a_last_sentence_remark(assistant, i)
        do_a_dummy_agree()
        time.sleep(random.uniform(0.5, 2))



# if user input sth
if input_text:
    # add signs
    # input_text = add_sign_input(input_text) 
    st.success('Message Send Successfully! Do not trust anyone in the room!', icon='✅')
    st.session_state.chat_history.append({"type":'warning', "name":None, "text":None, "avatar":None, "quote":None})

    with st.chat_message("user", avatar='🤔'): 
        st.markdown(input_text) 
    st.session_state.chat_history.append({"type":'message', "name":"user", "text":input_text, "avatar":'🤔', "quote":None}) 
    
    # generate response randomly
    time.sleep(1.5)
    assistant = random.choice(assistants)
    do_a_direct_response(input_text, assistant)
    do_a_dummy_agree()
    
    # st.write(f"Response generated in {elapsed_time:.2f} seconds.")

    # generate remarks randomly
    times = random.randint(2, 3)
    for i in range(times):
        assistant = random.choice(assistants)

        # functions = [
        #     lambda: do_a_direct_response(input_text, assistant),
        #     lambda: do_a_last_sentence_remark(assistant)
        # ]
        # p = [0.2, 0.8]
        # selected_callable = random.choices(functions, p, k=1)[0]
        # selected_callable()
        do_a_last_sentence_remark(assistant, i)
        time.sleep(random.uniform(0.5, 1))
        do_a_dummy_agree()

    # generate response randomly
    time.sleep(1)
    assistant = random.choice(assistants)
    do_a_direct_response(input_text, assistant)
    do_a_dummy_agree()

    # generate funny remarks
    p = random.uniform(0, 1)
    if p > 0.5:
        assistant = random.choice(assistants)
        do_a_funny_remark(assistant)

    # generate remarks randomly
    times = random.randint(2, 3)
    for i in range(times):
        assistant = random.choice(assistants)

        # functions = [
        #     lambda: do_a_direct_response(input_text, assistant),
        #     lambda: do_a_last_sentence_remark(assistant)
        # ]
        # p = [0.2, 0.8]
        # selected_callable = random.choices(functions, p, k=1)[0]
        # selected_callable()
        do_a_last_sentence_remark(assistant, i)
        do_a_dummy_agree()
        time.sleep(random.uniform(0.5, 1))
        

    
        