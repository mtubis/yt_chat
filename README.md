# yt_chat
chat with YouTube movie

## Dependences:

```
pip install openai==1.12.0 langchain==0.1.7 langchain-openai==0.0.6 chromadb beautifulsoup4==4.12.3 streamlit youtube-transcript-api
```

## Task:

Implement, using the streamlit library, a web application that allows users to ask questions about the content of YouTube videos.

The application should allow the user to enter the URLs of a YouTube video and then display a chat window where the user can ask questions about the content of the video.
The application should use a vector database to search for text snippets and a GPT-3 language model to generate answers to questions. The answers should be displayed in the chat window.

### Tips:

https://docs.streamlit.io/knowledge-base/tutorials/build-conversational-apps
https://docs.streamlit.io/library/api-reference/status
https://python.langchain.com/docs/integrations/document_loaders/youtube_transcript

The search can be implemented as a function triggered by OpenAI Functions and, for example, forced to be triggered at the start of a conversation via the parameter tool_choice={"type": "function", "function": {"name": "search"}}

## Running:

```
streamlit run app.py
```
