from transformers import BertTokenizer, BertForSequenceClassification
import torch
import os 
from dotenv import load_dotenv

load_dotenv()
# Load the tokenizer and model from the local directory
tokenizer = BertTokenizer.from_pretrained("./finetuned bert")
model = BertForSequenceClassification.from_pretrained("./finetuned bert")

# Function to classify user input
def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class

# Example of conversation
user_input = "I can't sleep properly nowadays."
status_mapping={0: "Anxiety", 1: "Normal",2: "Depression",3: "Suicidal",4: "Stress",5: "Bipolar",6: "Personality disorder"}

predicted_status = classify_text(user_input)
print(f"Predicted mental health status: {status_mapping[predicted_status]}")


from src.prompt import *
#from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
#from langchain_core.output_parsers import StrOutputParser



#prompt=ChatPromptTemplate.from_template(prompt_template)

GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"]=GOOGLE_API_KEY

llm=ChatGoogleGenerativeAI(model='gemini-1.5-pro')


def get_prompt(predicted_status, user_feelings):
    # Add a default value for predicted_status
    predicted_status = predicted_status if predicted_status is not None else 0
    
    return f"""
        You are an empathetic mental health chatbot 
        designed to support users based on their current emotional and mental health status.
        You are an empathetic mental health chatbot. 
        The user has expressed that they are feeling {status_mapping[predicted_status]} 
        and shared the following details about how they are feeling: "{user_feelings}".
        
        Your task is to respond with kindness.
    """

# Define user_feelings
user_feelings = "I can't sleep properly nowadays."

# Use RunnablePassthrough to wrap the get_prompt function
prompt_runnable = RunnablePassthrough(func=lambda x: get_prompt(predicted_status, x))

# Use the prompt_runnable in the chain
chain = (
    prompt_runnable
    | llm
    #| StrOutputParser()
)

result = chain.invoke(input=user_feelings)

print(result.content)


