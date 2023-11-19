import os
import glob

from typing import Optional, List
from enum import Enum

from langchain.pydantic_v1 import BaseModel, Field
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains.openai_functions import (
    convert_to_openai_function,
    get_openai_output_parser,
)

class ToneEnum(str, Enum):
    positive = "positive"
    negative = "negative"


class Email(BaseModel):
    """Relevant information about an email."""

    sender: Optional[str] = Field(None, description="The sender's name, if available")
    sender_phone_number: Optional[str] = Field(None, description="The sender's phone number, if available")
    sender_address: Optional[str] = Field(None, description="The sender's address, if available")
    action_items: List[str] = Field(..., description="A list of action items requested by the email")
    topic: str = Field(..., description="High level description of what the email is about")
    tone: ToneEnum = Field(..., description="The tone of the email.")
    

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert researcher."),
        (
            "human",
            "What can you tell me about the following email? Make sure to answer in the correct format: {email}",
        ),
    ]
)

openai_functions = [convert_to_openai_function(Email)]
llm_kwargs = {
    "functions": openai_functions,
    "function_call": {"name": openai_functions[0]["name"]}
}

llm = ChatOpenAI(temperature=0, model="gpt-4-1106-preview")

output_parser = get_openai_output_parser([Email])
extraction_chain = prompt | llm.bind(**llm_kwargs) | output_parser

files = glob.glob('./dataset/*')

for file in files:
    with open(file, 'r') as f:
        content = f.read()
        print(file)
        extraction_chain.invoke({
            "email": content
        })
