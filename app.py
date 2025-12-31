from fastapi import FastAPI
from pydantic import BaseModel
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
from dotenv import load_dotenv
import uvicorn
load_dotenv()
# ------------- State -------------
class State(TypedDict):
    messages: Annotated[list[dict], add_messages]
# ------------- LLM -------------
# Ensure the API Key is set. 
# Ideally, this should be in a .env file: GROQ_API_KEY=gsk_...
# But we fallback to the one provided if not in env.
if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = "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"
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    # api_key is automatically read from os.environ["GROQ_API_KEY"]
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         """You are an AI expense assistant specifically designed to help students manage their finances.
         
You have access to the user's actual expense data from their database. When answering questions:
- Use the provided USER DATA to give accurate, personalized responses
- Be specific with numbers and categories
- Provide actionable budget advice based on their spending patterns
- Highlight areas where they're overspending
- Celebrate good financial habits
- Be friendly, supportive, and encouraging

When user data is provided in the context, always reference their actual numbers rather than asking for information."""),
        MessagesPlaceholder(variable_name="messages")
    ]
)
chain = prompt | llm
app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ------------- Node -------------
def guide_node(state: State):
    response = chain.invoke({"messages": state["messages"]})
    return {"messages": [response]}
# ------------- LangGraph -------------
builder = StateGraph(State)
builder.add_node("guide", guide_node)
builder.add_edge(START, "guide")
builder.add_edge("guide", END)
graph = builder.compile()
class ChatRequest(BaseModel):
    history: list
    user_input: str
    context: str = ""
@app.post("/chat")
def chat_endpoint(data: ChatRequest):
    messages = data.history
    # If context is provided, combine it with the user's message
    user_message = data.user_input
    if data.context:
        # Context includes the user's question at the end, so use it directly
        user_message = data.context
    messages.append({"role": "user", "content": user_message})
    
    result = graph.invoke({"messages": messages})
    last_message = result["messages"][-1]
    bot_reply = last_message.content
    def msg_to_dict(m):
        if hasattr(m, 'type'):
            role = "assistant" if m.type == "ai" else "user"
            return {"role": role, "content": m.content}
        return m
    updated_history = [msg_to_dict(m) for m in result["messages"]]
    return {"reply": bot_reply, "history": updated_history}
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)