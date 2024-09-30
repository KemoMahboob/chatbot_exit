import os
from flask import Flask, render_template, request, jsonify
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain import LLMChain
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

app = Flask(__name__)

# Initialize ChatGroq LLM
groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key is None:
    raise ValueError("GROQ_API_KEY environment variable not set")

llm = ChatGroq(
    model='llama3-70b-8192',
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=groq_api_key
)

# Function to generate a traffic report based on the CSV data
def generate_traffic_report():
    try:
        file_path = os.path.join('assets', 'Exit 7 .csv')  # Ensure there are no spaces in the filename
        if not os.path.isfile(file_path):
            return "The traffic data file is not found."
        
        df = pd.read_csv(file_path)
        latest_data = df.tail(1).to_dict(orient='records')[0]
        
        exit_name = latest_data.get('Exit')
        location = latest_data.get('Location')
        street = latest_data.get('Street')
        traffic_state = latest_data.get('Traffic State')
        gate_state = latest_data.get('Gate State')
        timestamp = latest_data.get('Timestamp')
        
        report = f"""
        ### Traffic Report for {exit_name}
        
        **Exit Name:** {exit_name}  
        **Traffic State:** {traffic_state}  
        **Location:** {location}  
        **Street:** {street}  
        **Gate State:** {gate_state}  
        **Time:** {timestamp}
        
        Based on current traffic, you may need to close the gate if conditions worsen.
        """
        return report
    except Exception as e:
        return f"Error reading the data: {str(e)}"

# Define prompt template for the chatbot
template = """
### Instructions:
You are an AI assistant designed to provide traffic management information in a conversational and helpful way, similar to ChatGPT. Your responses should be friendly, clear, and focused on the user's request.

*Use the provided traffic data to give an accurate, concise, and helpful response.* If the data is unclear or missing, explain that to the user and offer suggestions if needed. Always aim to guide the user in a conversational and supportive manner.

### Traffic Data:
{context}

### Question: {question}
---
### Response:
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)

chain = LLMChain(llm=llm, prompt=prompt)

@app.route('/')
def home():
    return '''
    <h1>Welcome to Traffic Management System</h1>
    <a href="/dashboard"><button>Go to Dashboard</button></a>
    <a href="/chatbot"><button>Go to Chatbot</button></a>
    '''

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/generate_report', methods=['POST'])
def generate_report():
    user_question = request.form.get("question")  # Changed from request.json to request.form
    context = generate_traffic_report()
    
    # Generate response using the LLM chain
    response = chain.run(context=context, question=user_question)
    return jsonify({"response": response})

@app.route('/traffic_data')
def traffic_data():
    file_path = os.path.join('assets', 'Exit 7 .csv')  # Ensure there are no spaces in the filename
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        traffic_distribution = df['Traffic State'].value_counts().to_dict()
        return jsonify({
            "data": df.to_dict(orient='records'),
            "traffic_distribution": traffic_distribution
        })
    else:
        return jsonify({"error": "The traffic data file is not found."}), 404

if __name__ == '__main__':
    app.run(debug=True)
