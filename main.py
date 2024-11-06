from fastapi import FastAPI
from pydantic import BaseModel
import os
import uuid

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from typing import Optional

# Load environment variables for authentication
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Mount the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open("index.html", "r", encoding='utf-8') as f:
        return f.read()


# Define the persistent directory for vector store and load it
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# Check if the Chroma vector store already exists
if os.path.exists(persistent_directory):
    print("Loading existing vector store...")
    db = Chroma(persist_directory=persistent_directory,
                embedding_function=None)
else:
    raise FileNotFoundError(
        f"The directory {persistent_directory} does not exist. Please check the path."
    )

# Define the embedding model (Basque embeddings from your dialogues)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Load the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Create a retriever for querying the vector store
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},  # Retrieve top 3 relevant dialogues
)

# Create a ChatOpenAI model (Basque support via GPT)
llm = ChatOpenAI(model="gpt-4o")

# Contextualize question prompt for chat history rephrasing
# You solely act, think and write in Basque. Given the chat history and last user question, if there is referable context in that history, formulate an autonomous question in Basque, which can be understood even without chat history. DO NOT answer the question.
contextualize_q_system_prompt = (
    "Euskaraz jarduten, pentsatu eta idazten duzu soilik.Txat-historia eta erabiltzailearen azken galdera kontuan hartuta, historia horretan erreferentziazko testuingurua badago, galdera autonomo bat formulatu euskaraz, txat-historiarik gabe ere uler daitekeena.EZ erantzun galderari.")

# Create a prompt template for contextualizing questions
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a history-aware retriever
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# QA prompt for answering based on retrieved context
# You are an assistant who only communicates in Basque, and you will never respond in another language.
#  "You have improved knowledge of natural language and grammar gained from film interviews. Always answer in Basque, "
#  "short and clear. You don't know any other language, you only know Basque.\n\n{context}"
qa_system_prompt = (
    "Euskaraz soilik komunikatzen duen laguntzaile bat zara, eta ez duzu inoiz erantzungo beste hizkuntza batean. Filmetako elkarrizketetatik lortutako hizkuntza naturala eta gramatika ezagutza hobetuak dituzu. Beti euskaraz erantzun, labur eta argi. Ez duzu beste hizkuntzarik ezagutzen, euskaraz soilik dakizu.\n\n{context}"
)

# Create a prompt template for answering questions
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a chain to combine retrieved documents for question answering
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Create a retrieval chain that combines the history-aware retriever and question-answering chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Set up the ReAct agent with the document store retriever
react_docstore_prompt = hub.pull("hwchase17/react")

tools = [
    Tool(
        name="Answer Question",
        func=lambda input, **kwargs: rag_chain.invoke(
            {"input": input, "chat_history": kwargs.get("chat_history", [])}
        ),
        description="Erabilgarria testuinguruari buruzko galderak erantzun behar dituzunean.",
    )
]

# Create the ReAct Agent with document store retriever
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_docstore_prompt,
)

# Create the agent executor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, handle_parsing_errors=True, verbose=True
)

# Create a dictionary to store chat histories per session
chat_histories = {}


# Define Pydantic models for request and response
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    user_message = request.message
    session_id = request.session_id

    # If session_id is not provided, create a new one
    if not session_id:
        session_id = str(uuid.uuid4())
        # Initialize chat history for the new session
        chat_histories[session_id] = [
            AIMessage(
                content=(
                    # "I am an assistant for Basque speaking, with enhanced knowledge in grammar and natural language from dialogues from movie subtitles.
                    "Euskararen laguntzailea naiz, pelikulen azpitituluetako elkarrizketetatik abiatuta gramatikaren eta hizkuntza naturalaren ezagutza hobetua dut"
                )
            )
        ]

    # Get the chat history for this session
    chat_history = chat_histories.get(session_id, [])
    # Append user's message to chat history
    chat_history.append(HumanMessage(content=user_message))
    # Get response from agent
    response = agent_executor.invoke(
        {"input": user_message, "chat_history": chat_history}
    )
    ai_message = response["output"]
    # Append AI's response to chat history
    chat_history.append(AIMessage(content=ai_message))
    # Update the chat history
    chat_histories[session_id] = chat_history
    # Return response and session_id to user
    return ChatResponse(response=ai_message, session_id=session_id)