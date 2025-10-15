import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.agents import Tool
from langchain.prompts import PromptTemplate

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPEN_ROUTER_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ['OPENAI_API_BASE'] = 'https://openrouter.ai/api/v1'
os.environ['OPENAI_BASE_URL'] = 'https://openrouter.ai/api/v1'

def search_web(query: str) -> str:
    """Search the web for information"""
    try:
        search = DuckDuckGoSearchAPIWrapper()
        results = search.run(query)
        return f"Web search results:\n{results}"
    except Exception as e:
        return f"Error performing web search: {str(e)}"

# Define a tool the agent can use
tools = [
    Tool(
        name="PDF_Reader",
        func=lambda file_path: "C:\GenAI\GenAIEngineering-Cohort2\Week13\Day_2\LLM Introduction.pdf".join(FAISS.from_texts(
            RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0).split_texts(
                PyMuPDFLoader(file_path).load_and_split()
            ),
            OpenAIEmbeddings()
        ).similarity_search("What is the main topic discussed in the document?", k=1)[0].page_content),
        description="Useful for reading and extracting text from PDF documents"
    ),
    Tool(
        name="Web_Search",
        func=search_web,
        description="Search the internet for information. Use this when the PDF doesn't contain the answer or when you need additional context."
    )
]

# Create a chat model
llm = ChatGoogleGenerativeAI(
        model='gemini-1.5-flash',
        api_key=os.getenv('GOOGLE_API_KEY'),
    )


# Create a prompt template for ReAct agent
react_prompt = PromptTemplate.from_template(
    """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

IMPORTANT: 
1. Always try PDF_Search first
2. If PDF_Search returns "No relevant information found" or insufficient information, use Web_Search
3. Combine information from both sources when relevant

Begin!

Question: {input}
Thought: {agent_scratchpad}"""
)

# Create a ReAct agent
agent = create_react_agent(llm, tools, prompt=react_prompt)  

# Wrap in an executor to run
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Run agent
if __name__ == "__main__":
    query = "What is the main topic discussed in the document?"
    print(f"\nQuery: {query}\n")
    
    response = executor.invoke({"input": query})
    print(f"\n{'='*50}")
    print(f"Final Answer: {response['output']}")
    print(f"{'='*50}")    