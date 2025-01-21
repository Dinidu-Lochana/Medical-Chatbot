import os
import time
import streamlit as st
from typing import Sequence
from typing_extensions import Annotated, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    trim_messages,
    BaseMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from dotenv import load_dotenv


load_dotenv()


class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], list]
    language: str


# AI Model
class Model:
    def __init__(self):
        self.model = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        self.prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a friendly and knowledgeable health companion. Always introduce yourself as Baymax when someone greets you with phrases like 'Who are you?', 'Hello', or 'Hi'. Respond with: 'Hello, I am Baymax. Your personal healthcare companion.' for such greetings. "
                "If someone thanks you, reply with: 'You have been a good boy. Have a lollipop.' For all other health-related queries, provide accurate, supportive, and helpful advice in {language}.",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
        )

        self.trimmer = trim_messages(
            max_tokens=65,
            strategy="last",
            token_counter=self.model,
            include_system=True,
            allow_partial=False,
            start_on="human",
        )

    def invoke(self, query, chat_message_history):
        trimmed_messages = self.trimmer.invoke(chat_message_history.messages + [HumanMessage(query)])
        prompt = self.prompt_template.invoke({"messages": trimmed_messages, "language": "en"})
        response = self.model.invoke(prompt)
        chat_message_history.add_user_message(query)
        chat_message_history.add_ai_message(response.content)
        return response.content


# Chat Interface
class ChatInterface:
    def __init__(self):
        self.model = Model()

    def get_chat_history(self, user_id):
        return MongoDBChatMessageHistory(
            session_id=user_id,
            connection_string=os.getenv("MONGO_DB_URI"),
            database_name="Chatbot",
            collection_name="chat_histories",
        )

    def chat(self, message, history, user_id, language="English"):
        try:
            if not user_id:
                return "", history, "Please enter a user ID first."

            chat_history = self.get_chat_history(user_id)
            with st.spinner("BayMax is typing..."):
                response = self.model.invoke(message, chat_history)

            
            placeholder = st.empty()
            full_response = ""
            for char in response:
                full_response += char
                placeholder.markdown(f"**BayMax:** {full_response}")
                time.sleep(0.03)  

            return "", history + [(message, response)], ""
        except Exception as e:
            return "", history, f"Error: {str(e)}"

    def reset_chat(self, user_id):
        try:
            if not user_id:
                return None, "Please enter a user ID first"
            chat_history = self.get_chat_history(user_id)
            chat_history.clear()
            return None, f"Chat history cleared for user {user_id}"
        except Exception as e:
            return None, f"Error clearing chat: {str(e)}"



# Streamlit UI
def main():
    st.set_page_config(page_title="BayMax - Medical Chatbot", layout="wide")
    

    st.markdown(
        """
        <style>
        .st-chat {
            max-width: 800px !important;
            margin: auto !important;
            padding: 20px !important;
            background-color: #f7f7f7 !important;
            border-radius: 10px !important;
        }
        .st-chat-user {
            text-align: right;  /* Align user messages to the right */
            color: white !important;
            font-weight: bold;
        }
        .st-chat-bot {
            text-align: left;  /* Align bot messages to the left */
            color: #ff6f61 !important;
            font-weight: bold;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🤖 BayMax")
   

    chat_interface = ChatInterface()

    
    with st.sidebar:
        st.header("Settings")
        user_id = st.text_input("Username:", value="user1", placeholder="Enter your User ID")
        language = st.selectbox("Language", ["English", "Spanish", "French", "German", "Chinese"])

        
        if st.button("🗑️ Clear Chat"):
            _, status = chat_interface.reset_chat(user_id)
            st.success(status)

    # chat history
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

   
    chat_placeholder = st.container()  

    # Input Section
    
    message = st.text_input("Your message:", placeholder="Ask BayMax something...")
    send_button = st.button("🚀 Send")

    if send_button and message.strip():
        
        _, st.session_state["chat_history"], error = chat_interface.chat(
            message, st.session_state["chat_history"], user_id, language
        )
        if error:
            st.error(error)

    
    with chat_placeholder:
        for user, msg in st.session_state["chat_history"]:
            if user:  
                st.markdown(
                    f"""
                    <div class="st-chat-user">
                        You: {user}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown(
                f"""
                <div class="st-chat-bot">
                    BayMax: {msg}
                </div>
                """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
