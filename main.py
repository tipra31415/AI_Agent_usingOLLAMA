from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool


load_dotenv()

class ResearchResponse(BaseModel):
    topic:str
    summary:str
    sources:list[str]
    tools_used:list[str]


llm = ChatOllama(model="llama3.2:1b")
parser=PydanticOutputParser(pydantic_object=ResearchResponse)

prompt=ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and necessary tools.messages
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder","{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}")
    ]
).partial(format_instructions=parser.get_format_instructions())

tools=[search_tool]
agent=create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools,
)

agent_executor=AgentExecutor(agent=agent,tools=tools,verbose=True)
query=input("What can i help you research?")
# In your main.py, replace the parsing section with:
raw_response = agent_executor.invoke({"query": query})


try:
    structured_response=parser.parse(raw_response.get("output")["text"])
    print(structured_response)
except Exception as e:
    print("Error parsing response",e,"Raw Response - ",raw_response)

