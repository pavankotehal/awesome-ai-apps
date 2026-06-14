"""LangGraph starter — a prebuilt ReAct agent powered by Nebius via LiteLLM."""
import os
from datetime import datetime
import json

from dotenv import load_dotenv
from langchain_litellm import ChatLiteLLM
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
# from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent
from tools import search_tool, wiki_tool, save_tool


load_dotenv()

# # simple class to parse the LLM response into a structured format
# class LLMResponse(BaseModel):
#     user_query: str = Field(description="The original user query that was classified.")
#     sentiment: str = Field(description="The sentiment classification: 'neutral', 'negative', or 'positive'.")

class ResearchResponse(BaseModel):
    topic: str = Field(description="The main topic of the research query.")
    summary: str = Field(description="A brief summary of the research query.")
    sources: list[str] = Field(description="A list of sources used for the research.")
    tools_used: list[str] = Field(description="A list of tools used during the research process.")
    findings: str = Field(description="A summary of the research findings related to the query.")


# initialize the LLM with the Nebius API key from the environment variables
llm = ChatLiteLLM(
        model="openai/deepseek-ai/DeepSeek-V3.2",
        api_base="https://api.tokenfactory.nebius.com/v1/",
        api_key=os.environ["NEBIUS_API_KEY"],
    )

structured_llm = llm.with_structured_output(ResearchResponse)

user_input = input("Enter a research query: ")


prompt = ChatPromptTemplate.from_messages([
    (
        "system", "You are a helpful assistant that researches, "
        "and uses tools when needed to answer the user's query. "
        "Always use the tools at your disposal if they can help you answer the question more accurately or efficiently. "
        "After researching, provide a structured response with the main topic, "
        "a brief summary, sources used, tools used, and your findings."
        ),
        (
            "user", "Research the following topic: {user_input}"
        ),
    
])



# define the chain
chain = prompt | structured_llm
response = chain.invoke({"user_input": user_input})
print(response)

# Save to JSON in the same directory
output_path = os.path.join(os.path.dirname(__file__), "results.json")

if os.path.exists(output_path):
    with open(output_path, "r") as f:
        results = json.load(f)
else:
    results = []

results.append(response.model_dump())

with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"Saved to {output_path}")