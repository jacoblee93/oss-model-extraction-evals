# NOTES: Include DEFINED instructions and Train on things that it can't understand on its own
#        And includes a function to help with names & nouns
#        And run with a local model
#        TODO: Run with local embeddings
#        TODO: Feed it better rows, not the first 3
import os
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI

from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
from langchain.agents.agent_toolkits import create_retriever_tool

import ast
import re

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "ls__d48957b36ce548a78e251b485b0f3d29"

from openai import OpenAI

# For embeddings model, the example uses a sentence-transformers model
# https://www.sbert.net/docs/pretrained_models.html 
# "The all-mpnet-base-v2 model provides the best quality, while all-MiniLM-L6-v2 is 5 times faster and still offers good quality."
#embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME", "all-MiniLM-L6-v2")
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME", "all-mpnet-base-v2")

#from constants import CHROMA_SETTINGS

model = os.environ.get("MODEL", "mistral")
llm = OpenAI(base_url="http://localhost:11434/v1", api_key="not-needed")

# Initialize the OpenAI client with your API details
#llm = ChatOpenAI(model="gpt-4-0125-preview", temperature=0)
#engine = create_engine("sqlite:///GContacts.db")
#db = SQLDatabase(engine=engine)
db = SQLDatabase.from_uri(
	"sqlite:///GContacts.db",
	include_tables=['contacts'], # including only one table for illustration
	sample_rows_in_table_info=3
)
print(db.table_info)

# Function to help with proper nouns
def query_as_list(db, query):
    res = db.run(query)
    res = [el for sub in ast.literal_eval(res) for el in sub if el]
    res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]
    return list(set(res))

contact_names = query_as_list(db, "SELECT Name FROM contacts")

vector_db = FAISS.from_texts(contact_names, OpenAIEmbeddings())
retriever = vector_db.as_retriever(search_kwargs={"k": 5})
description = """Use to look up values to filter on. Input is an approximate spelling of the proper noun, output is \
valid proper nouns. Use the noun most similar to the search."""
retriever_tool = create_retriever_tool(
    retriever,
    name="search_proper_nouns",
    description=description,
)


# IMPROV: Using a dynamic few-shot prompt
examples = [
# Familiarity
    {
        "input": "Find contacts who I know really well",
        "query": "select * from contacts where Familiarity= '⭐️⭐️⭐️⭐️'", 
    },
    {
        "input": "Find contacts who I know pretty well",
        "query": "select * from contacts where Familiarity= '⭐️⭐️⭐️'", 
    },
    {
        "input": "Find contacts who I know a little",
        "query": "select * from contacts where Familiarity= '⭐️⭐️'", 
    },
    {
        "input": "Find contacts who are acquaintainces",
        "query": "select * from contacts where Familiarity= '⭐️'", 
    },
# Name & kids should be a like search with other columns
    {
        "input": "List all tracks in the 'Rock' genre.",
        "query": "SELECT Child, Son, Daughter FROM contacts WHERE Name like '%Amal%'",
    },
# Introduction
    {
        "input": "Who introduced me to Novo?",
        "query": "select name, Referredby from contacts where name like '%Novo%'",
    },
# Dates
    {
        "input": "What important dates are coming up in the next 7 days?",
        #"query": "SELECT Name, Birthday, Anniversary, \"Check In\" FROM contacts WHERE  (Birthday IS NOT NULL AND strftime('%m-%d', Birthday) BETWEEN strftime('%m-%d', 'now') AND strftime('%m-%d', 'now', '+7 days') )  OR (Anniversary IS NOT NULL AND strftime('%m-%d', Anniversary) BETWEEN strftime('%m-%d', 'now') AND strftime('%m-%d', 'now', '+7 days') )  OR (\"Check In\" IS NOT NULL AND strftime('%m-%d', \"Check In\") BETWEEN strftime('%m-%d', 'now') AND strftime('%m-%d', 'now', '+7 days') )  ORDER BY Name",
        "query": "SELECT Name, Birthday, Anniversary, \"Check In\" FROM contacts WHERE (Birthday IS NOT NULL AND strftime('%m-%d', Birthday) BETWEEN strftime('%m-%d', 'now') AND strftime('%m-%d', 'now', '+7 days')) OR (Anniversary IS NOT NULL AND strftime('%m-%d', Anniversary) BETWEEN strftime('%m-%d', 'now') AND strftime('%m-%d', 'now', '+7 days')) OR (\"Check In\" IS NOT NULL AND \"Check In\" BETWEEN date('now') AND date('now', '+7 days')) ORDER BY Name",
    },
    {
        "input": "Who have I engaged with in the last year?",
        "query": "SELECT Name, \"Connected On\", \"Last Connected\", \"Last Meet\" FROM contacts WHERE (\"Connected On\" IS NOT NULL AND \"Connected On\" BETWEEN date('now', '-365 days') AND date('now')) OR (\"Last Connected\" IS NOT NULL AND \"Last Connected\" BETWEEN date('now') AND date('now', '-365 days')) OR (\"Last Meet\" IS NOT NULL AND \"Last Meet\" BETWEEN date('now') AND date('now', '-365 days')) ORDER BY Name",
    },
# Interests
# Data Quality
]


example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(),
    FAISS,
    k=5,
    input_keys=["input"],
)

from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)

system_prefix = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the given tools. Only use the information returned by the tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.
If you are querying for dates and you get multiple columns which have older dates, don't mention those columns in your respnse.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

If you need to filter on a proper noun, you must ALWAYS first look up the filter value using the "search_proper_nouns" tool! 
You have access to the following tables: contacts

If the question does not seem related to the database, just return "I don't know" as the answer.

Here are some examples of user inputs and their corresponding SQL queries:"""

few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=PromptTemplate.from_template(
        "User input: {input}\nSQL query: {query}"
    ),
    input_variables=["input", "dialect", "top_k"],
    prefix=system_prefix,
    suffix="",
)

full_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate(prompt=few_shot_prompt),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

# Example formatted prompt
prompt_val = full_prompt.invoke(
    {
        "input": "Who do I know really well that likes golf?",
        "top_k": 5,
        "dialect": "SQLite",
        "agent_scratchpad": [],
    }
)
print(prompt_val.to_string())

agent = create_sql_agent(
    llm=llm,
    db=db,
    prompt=full_prompt,
    verbose=True,
    agent_type="openai-tools",
)

#agent.invoke({"input": "Who do I know really well that likes golf?"})
#agent.invoke({"input": "Who introduced me to Novo?"})
#agent.invoke({"input": "What important dates are coming up in the next 30 days?"})
#agent.invoke({"input": "Who have I engaged with in the last month?"})

prompt = ChatPromptTemplate.from_messages(
    [("system", system_prefix), ("human", "{input}"), MessagesPlaceholder("agent_scratchpad")]
)
agent = create_sql_agent(
    llm=llm,
    db=db,
    extra_tools=[retriever_tool],
    prompt=prompt,
    agent_type="openai-tools",
    verbose=True,
)

agent.invoke({"input": "What are Jaiandran's interests?"})