import streamlit as st
import sys

device = 'cpu'
a=0


#===========================================#
#        Loads Model and word_to_id         #
#===========================================#
@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_model():
    from transformers import GPTNeoForCausalLM, GPT2Tokenizer, GPT2ModelForCausalLM
    print('loading model')
    sys.stdout.flush()
    #model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")
    #tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
    model = GPT2ModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("sshleifer/tiny-gpt2")

    model.to(device)
    return tokenizer, model




#===========================================#
#              Streamlit Code               #
#===========================================#

if __name__ == "__main__":
    seed_options = ['''Tweet: I thought I had heard it all. But, no. Now the #Vatican and #Maduro are in bed together, praising #Castro’s murderous communist regime on the 60th yr of Fidel’s #Cuban Revolution.''',
         '''Tweet: Is the #US #MIC Plotting WAR with #Venezuela? All the #MSMLies are Anti #Maduro Yet #Citizens In Venezuela Fully Support his presidency. What the MSM won't tell #WeThePeople''',
         '''Tweet: LOL! #Maduro’s government actually IS legitimate UNLIKE our current cabal here in the banana republic United States. Latin America has had enough. #HandsOffVenezuela #Venezuela''',
         '''Tweet: When Donald Trump talked to Wagner College students in 2004, walls were a bad thing. Rt about your fucking wall and #TrumpShutdown #shutdown''',
         '''Tweet: Today, the Dictator has been “sworn” for another 6 years. We protest again the dictatorship!!! Venezuela libre!!!''',
        '''Tweet: New Year's Eve! Goddaughter! 🤘🏼😍🤘🏼 #newyearseve #chilling #worldwide''',
        '''Tweet: Venezuelans have had to survive and not survive with #Chavez and #Maduro. It breaks my heart to listen how my family is struggling. Apart from sending my cousins food, medicine and clothes, what else can I do? 💔💔💔💔💔''',
        '''Tweet: Informal.
In police groups they manage to filter a photograph of Óscar Pérez killed and neutralized by military and police forces of the Nicolás Maduro regime.
11:32 pm - Even the Government of Venezuela has not made the news official to the media.''',
        '''Tweet: Óscar Pérez, a rebel leader in #Venezuela, was killed six months ago in a siege at his Caracas hideout. We show how the day unfolded and why our investigation concludes that he Pérez was possibly executed.''',
        '''Tweet: Outside perspective of the battle today in #Venezuela, as Oscar Pérez and his group came under siege by police when they discovered the hideout. Sounds like a grenade of some kind was used. RPG? Still no confirmation on Pérez being dead or alive.''',
        '''Tweet: Footage of the armoury raid in #Venezuela. It seems Oscar Perez has some how managed to still evade capture after his helicopter stunt and is leading this group.''',
        '''Tweet: Maduro, referring to Oscar Perez (@EquilibrioGV) and his rebel soldiers as "terrorists", confirms the weapons raid on a National Guard unit in #Venezuela.''',
]

    question_options = [
        '''The agenda behind this tweet is to''',
        '''The emotion of this tweet is''',
        '''The author of this tweet believes that Maduro''',
        '''The author of this tweet believes that Venezuela''',
        '''The author of this tweet believes that Oscar Perez''',
    ]

    tokenizer, model = load_model()

    st.title('GPT-Neo')

    response_max_length = st.sidebar.number_input('Response max length', min_value=1, max_value=1000, value=50)
    num_beams = st.sidebar.number_input('num_beams', min_value=1, max_value=50, value=5)
    do_sample = st.sidebar.checkbox('do_sample', value=True)
    top_p = st.sidebar.slider('top_p', min_value=0., max_value=1., value=0.97)
    top_k = st.sidebar.number_input('top_k', min_value=1, max_value=500, value=100)
    temperature = st.sidebar.slider('temperature', min_value=0., max_value=3., value=1.)
    eos_token = st.sidebar.text_area('eos_token (When to stop generating. Default is two newline chars "\\n\\n")', value='\n\n')
    num_return_sequences = st.sidebar.number_input('num_return_sequences', min_value=1, max_value=10, value=1)
    num_beam_groups = st.sidebar.number_input('num_beam_groups', min_value=1, max_value=20, value=1)
    diversity_penalty = st.sidebar.slider('diversity_penalty', min_value=0., max_value=20., value=0.)
    option = st.selectbox(
        'Example seed texts:',
        seed_options
    )
    question_selectbox = st.selectbox(
        'Example questions:',
        question_options
    )
    user_input = st.text_area('Seed Text:', value=option + '\n\n' + question_selectbox, height=300)

    conf_dict = dict(num_beams=num_beams, do_sample=do_sample, top_p=top_p, top_k=top_k, temperature=temperature,
                     num_return_sequences=num_return_sequences, num_beam_groups=num_beam_groups, diversity_penalty=diversity_penalty)

    if st.button('Generate Text'):
        text = user_input
        input_ids = tokenizer(text, return_tensors="pt").input_ids
        input_ids = input_ids.to(device)
        input_ids_length = input_ids.shape[1]


        gen_tokens = model.generate(input_ids, max_length=input_ids_length+response_max_length, no_repeat_ngram_size=3,
                                    eos_token_id=tokenizer._convert_token_to_id(eos_token) if eos_token != '' else None, **conf_dict)
        output_texts = tokenizer.batch_decode(gen_tokens)
        for output_text in output_texts:
            final_text = output_text[len(text):]
            if eos_token != '':
                final_text = final_text.split(eos_token)[0]
            print('----------------------------------')
            print(text)
            print('--')
            print(final_text)
            sys.stdout.flush()
            # final_text = text
            st.write(final_text)
