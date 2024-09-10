import os

import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from openai import OpenAI

from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

st.title("ChatGPT-like clone")

os.environ["OPENAI_API_KEY"] = "put_openai_api_key_here"


def load_youtube_video(video_url):
    try:
        video_id = video_url.split("v=")[1]
        loader = YoutubeLoader(video_id, add_video_info=False)
        return loader.load()
    except IndexError as exc:
        raise IndexError(f"Invalid YouTube URL: `{video_url}` - ommiting...") from exc


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100
    )
    splits = []
    for doc in documents:
        splits += text_splitter.split_documents(doc)
    return splits


def store_videos_in_chroma():
    with st.spinner("Loading videos..."):
        videos_documents = []
        for url in st.session_state.urls:
            try:
                video_doc = load_youtube_video(url)
                videos_documents.append(video_doc)
            except IndexError as exc:
                st.warning(exc)

        splitted = split_documents(videos_documents)
        Chroma.from_documents(splitted, OpenAIEmbeddings(), persist_directory="./chroma_db")
        st.success("Videos loaded successfully. Write a question in chat window below.")
        st.session_state.videos_loaded = True


def format_docs(docs):
    return "\n\n".join(x.page_content for x in docs)


openai = ChatOpenAI(api_key=os.environ["OPENAI_API_KEY"], model="gpt-3.5-turbo")
prompt = ChatPromptTemplate.from_messages(
    [
        ("assistant", "You are a chatbot answering questions about YouTube videos."),
        ("assistant", "Given the following docs {docs_as_prompt} ask the question {question}"),
    ]
)


def call_ai(messages):
    retriever = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings()).as_retriever(
        search_type="similarity", search_kwargs={"k": 3})

    rag_chain = (
            {"docs_as_prompt": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | openai
            | StrOutputParser()
    )
    rag_stream = rag_chain.stream(messages[-1]["content"])
    resp = st.write_stream(rag_stream)
    return resp


# Set OpenAI API key from Streamlit secrets
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], )

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize YouTube URLs
if "urls" not in st.session_state:
    st.session_state.urls = []
    st.session_state.textarea = ""

# A flag to indicate if the app has saved the videos
if "videos_loaded" not in st.session_state:
    st.session_state.videos_loaded = False


def save_urls_in_session():
    st.session_state.urls = st.session_state.textarea.split("\n")


if not st.session_state.urls:
    st.text_area(label="Enter YouTube video URLs", key="textarea")
    st.button("Save", on_click=save_urls_in_session)


if st.session_state.urls:
    if not st.session_state.videos_loaded:
        store_videos_in_chroma()

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if question := st.chat_input("Ask a question about loaded YouTube videos..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": question})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(question)

    # Display assistant response in chat message container
        with st.chat_message("assistant"):
            response = call_ai(st.session_state.messages)
            st.session_state.messages.append({"role": "assistant", "content": response})
