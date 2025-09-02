import streamlit as st
import torch
from langchain.llms import ctransformers
from langchain.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer


#function to get response from llama 2 model
def getLlamaresponse(input_text,no_of_words,blog_style):
    model_name = "HuggingFaceH4/zephyr-7b-alpha"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name   # automatically use GPU if available
    )




    prompt = f"Write a blog for a topic {input_text} in {no_of_words} words for {blog_style}."

    # tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # generate response
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response







st.set_page_config(page_title='Generate Blogs',
                    page_icon= 'blog.png',
                    layout='centered',
                    initial_sidebar_state='collapsed')

st.header(f"Generate Blogs ")

input_text = st.text_input("Enter the topic for the blog")



col1,col2 = st.columns([5,5])


with col1:
    no_of_words = st.text_input("No of words")
with col2:
    blog_style = st.selectbox("Writing this blog for",('Reseachers','Students','General Public','Data Scientists'),index=0)
    
    
submit = st.button("Generate Blog")


#Final response

if submit:
    st.write(getLlamaresponse(input_text,no_of_words,blog_style))