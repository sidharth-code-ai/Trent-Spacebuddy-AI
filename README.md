# YouTube Video Chat Assistant

## Overview

An AI-powered assistant that allows users to ask questions and chat about YouTube videos by asking questions and receiving detailed answers based on the video transcript.  
Built using **LangChain**, **FAISS**, and **OpenAI GPT-4**, this AI system has a well-structured document pipeline.  
I have used **Streamlit** for making an interactive UI.  
Currently, this project is running locally, and I have not yet published it to the cloud. Since different cloud platforms offer various deployment methods, I am actively exploring the best options to host the application.

## Main Process
1. Takes in a YouTube URL.
2. Converts it into a full YouTube transcript.
3. Converts the document into vector embeddings and does a similarity search to return 4 individual chunks and generate a natural response (entire document pipeline).
4. Generates a response with memory as well.

## Installation
1. Clone the repository:  
   git clone https://github.com/your-username/repo-name.git

2.Install all the dependencies:
    pip install -r requirements.txt

3. Set yout Telegram Bot Token(Talk to Bot Father) and Open AI api Key:
    OPENAI_API_KEY="your_api_key"
    TELEGRAM_BOT_TOKEN="your_bot_token"

4. Run the Code 

Happy Coding, 
Thank you 

    
