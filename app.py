import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers

# Function to get response from LLaMA 2 model
def getLLamaResponse(input_text, no_words, blog_style):
    # Initialize the LLaMA 2 model with the correct configuration
    llm = CTransformers(model='models/llama-2-7b-chat.ggmlv3.q2_K.bin', 
                        model_type='llama',
                        config={'max_new_tokens': 256, 'temperature': 0.01})
    
    # Prompt template
    template = """
    Write a blog for {blog_style} job profile for a topic {input_text} within {no_words} words.
    """

    prompt = PromptTemplate(input_variables=["blog_style", "input_text", "no_words"],
                            template=template)

    # Generate the response from the LLaMA 2 model
    formatted_prompt = prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words)
    response = llm(formatted_prompt)
    
    return response

st.set_page_config(page_title="Generate Blogs",
                   layout='centered',
                   initial_sidebar_state='collapsed')  

st.header("Generate Blog")

input_text = st.text_input("Enter the blog topic")

# Creating columns for additional fields
col1, col2 = st.columns([5, 5])

with col1:
    no_words = st.text_input("Number of words")

with col2:
    blog_style = st.selectbox('Writing the blog for...',
                              ('Researcher', 'Data Scientist', 'Common people'), index=0)

submit = st.button("Generate")

# Final response
if submit:
    response = getLLamaResponse(input_text, no_words, blog_style)
    st.write(response)
