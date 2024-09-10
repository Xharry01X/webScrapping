from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage
import re

template = (
    "You are tasked with extracting specific information from the following text content: {dom_content}. "
    "Please follow these instructions carefully: \n\n"
    "1. **Extract Information:** Only extract the information that directly matches the provided description: {parse_description}. "
    "2. **No Extra Content:** Do not include any additional text, comments, or explanations in your response. "
    "3. **Empty Response:** If no information matches the description, return an empty string ('')."
    "4. **Direct Data Only:** Your output should contain only the data that is explicitly requested, with no other text."
)

model = OllamaLLM(model="llama2")  # Changed to llama2 as llama3.1 isn't a standard model name

def clean_location(location):
    # Remove any non-alphanumeric characters except spaces and commas
    cleaned = re.sub(r'[^a-zA-Z0-9,\s]', '', location)
    # Remove multiple spaces
    cleaned = re.sub(r'\s+', ' ', cleaned)
    # Trim leading and trailing whitespace
    return cleaned.strip()

def parse_with_ollama(dom_chunks, parse_description):
    prompt = ChatPromptTemplate.from_messages([
        ("human", template)
    ])
    chain = prompt | model
    
    parsed_results = []
    
    for i, chunk in enumerate(dom_chunks, start=1):
        try:
            response = chain.invoke(
                {"dom_content": chunk, "parse_description": parse_description}
            )
            
            if isinstance(response, AIMessage):
                result = clean_location(response.content)
            elif isinstance(response, str):
                result = clean_location(response)
            else:
                result = ''
            
            if result:  # Only append non-empty results
                parsed_results.append(result)
            
            print(f"Parsed batch: {i} of {len(dom_chunks)}")
        except Exception as e:
            print(f"Error parsing batch {i}: {str(e)}")
    
    return "\n".join(parsed_results)