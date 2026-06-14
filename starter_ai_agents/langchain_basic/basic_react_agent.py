"""ReAct agent powered by Nebius via LiteLLM."""
import os
import json

from dotenv import load_dotenv
from langchain_litellm import ChatLiteLLM
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langchain.agents import create_agent
# from tools import search_tool, wiki_tool, save_tool  # import your tools here

load_dotenv()

class ResearchResponse(BaseModel):
    topic: str = Field(description="The main topic of the research query.")
    summary: str = Field(description="A brief summary of the research query.")
    sources: list[str] = Field(description="A list of sources used for the research.")
    tools_used: list[str] = Field(description="A list of tools used during the research process.")
    findings: str = Field(description="A summary of the research findings related to the query.")


llm = ChatLiteLLM(
    model="openai/deepseek-ai/DeepSeek-V3.2",
    api_base="https://api.tokenfactory.nebius.com/v1/",
    api_key=os.environ["NEBIUS_API_KEY"],
)

tools = []  # add your tools here

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "You are a helpful assistant that researches and uses tools when needed. "
        "After researching, provide a structured response with the main topic, "
        "a brief summary, sources used, tools used, and your findings."
    ),
    response_format=ResearchResponse,
)

user_input = input("Enter a research query: ")

response = agent.invoke({"messages": [("user", user_input)]})

# response["structured_response"] is a ResearchResponse object when response_format is set
parsed = response["structured_response"]
print(parsed)

output_path = os.path.join(os.path.dirname(__file__), "results.json")

if os.path.exists(output_path):
    with open(output_path, "r") as f:
        results = json.load(f)
else:
    results = []

results.append(parsed.model_dump())

with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"Saved to {output_path}")
