from flask import Flask, render_template,request
from dotenv import load_dotenv
from transformers import BertTokenizer, BertForSequenceClassification
import torch
#from src.prompt import *
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
#from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
app=Flask(__name__)

load_dotenv()


#prompt=ChatPromptTemplate.from_template(prompt_template)


llm=ChatGoogleGenerativeAI(model='gemini-1.5-pro')
status_mapping={0: "Anxiety", 1: "Normal",2: "Depression",3: "Suicidal",4: "Stress",5: "Bipolar",6: "Personality disorder"}
def llm_chain(prompt_runnable):
    chain = (
    prompt_runnable
    | llm
    | StrOutputParser()
)

    return chain


tokenizer = BertTokenizer.from_pretrained("./finetuned bert")
model = BertForSequenceClassification.from_pretrained("./finetuned bert")

# Function to predict the mental status

def predict_mental_status(user_feelings):
    inputs=tokenizer(user_feelings, return_tensors='pt',padding=True,truncation=True)
    with torch.no_grad():
        outputs=model(**inputs)
    logits=outputs.logits
    predicted_class=torch.argmax(logits,dim=1).item()
    return predicted_class

def get_prompt(predicted_status, user_feelings):
    
    
    return f"""
        You are an empathetic mental health chatbot 
        designed to support users based on their current emotional and mental health status.
        You are an empathetic mental health chatbot. 
        The user has expressed that they are feeling {predicted_status} 
        and shared the following details about how they are feeling: "{user_feelings}".
        
        Your task is to respond with kindness.
    """
   
@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/get',methods=['GET','POST'])
def chat():
    # Bot starts the conversation
    initial_message = "Hello! I'm here to help you with anything you're feeling. How are you doing today?"
    print(f"Bot: {initial_message}")
    msg=request.form['msg']
    input=msg
    print(input)
    predicted_class=predict_mental_status(input)
    
    predicted_status=status_mapping[predicted_class]
    print(f"Predicted mental health status: {predicted_status}")
   
    prompt_runnable = lambda x: get_prompt(predicted_status, x)
    
    # Build the chain using the runnable prompt
    chain = llm_chain(prompt_runnable)
    result=chain.invoke(input=input)
    
    print(f"Bot: {result}")
    return result


if __name__=='__main__':
    app.run()






