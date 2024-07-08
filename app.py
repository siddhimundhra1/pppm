from flask import Flask, request, render_template_string, render_template, redirect, url_for
from openai import OpenAI
import mysql.connector
import re
import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np


# Initialize OpenAI client with your API key
client = OpenAI(api_key='sk-proj-LKBOtaN9R6rZQZTJV174T3BlbkFJcRypo2abZhqAHK9MiTbE')

app = Flask(__name__)

# Configure MySQL Database
db_config = {
    'database': 'chatgpt_responses',
    'host': 'localhost',
    'user': 'root'  
}

@app.route('/')
def index():
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM responses")
    entries = cursor.fetchall()
    cursor.execute("SELECT * FROM parsed_responses")
    parsed_entries = cursor.fetchall()
    cursor.close()
    connection.close()

    return render_template_string(open('index.html').read(), entries=entries, parsed_entries=parsed_entries)
    
 
@app.route('/process_input', methods=['POST'])
def process_input():
    user_input = request.form['user_input']

    prompt = (
      "Role: The categories within and beyond the organization that are granted access to private information. (ie Manager, Deliverer, Analyzer, Marketer, User, etc.) People who need the information. \n"
       "Purpose: reason data is needed \n"
       "Data attribute: pieces of sensitive information. Data item is an instance of a data attribute. (Attribute: age. Item: 26).\n"
       "Privacy Policy::=(role, purpose, data attribute) Remember these definitions. \n"
       "Convert this natural language into the privacy policy tuples. An example output can look like the following: (Company, to create an account, name) \n"
       "(Analyzer, to fight spam/malware, IP address)"
   )

    # Process input with ChatGPT
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
           {"role": "system", "content": "You extract (role, purpose, data attribute) tuples from the given privacy policy"},
           {"role": "user", "content": prompt + user_input}
        ]
    )

    chatgpt_response = completion.choices[0].message.content.strip()

    
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor()
    cursor.execute(
        "INSERT INTO responses (user_input, chatgpt_response) VALUES (%s, %s)",
       (user_input, chatgpt_response)
    )
    connection.commit()
    cursor.close()
    connection.close()
    return f"Stored response: {chatgpt_response}"

@app.route('/delete/<int:entry_id>', methods=['POST'])
def delete_entry(entry_id):
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor()
    query = "DELETE FROM responses WHERE id = %s"
    cursor.execute(query, (entry_id,))
    connection.commit()
    cursor.close()
    connection.close()
    
    return redirect('/')


@app.route('/parse/<int:entry_id>', methods=['POST'])
def parse_entry(entry_id):
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor()
    query = "SELECT chatgpt_response FROM responses WHERE id = %s"
    cursor.execute(query, (entry_id,))
    response = cursor.fetchone()[0]
    cursor.close()

    # Extract tuples from response text
    parsed_entries = re.findall(r'\(([^)]+)\)', response)
    parsed_entries = [tuple(entry.split(', ')) for entry in parsed_entries]

    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor()
    for role, purpose, data_attribute in parsed_entries:
        cursor.execute(
            "INSERT INTO parsed_responses (role, purpose, data_attribute, response_id) VALUES (%s, %s, %s, %s)",
            (role, purpose, data_attribute, entry_id)
        )
    connection.commit()
    cursor.close()
    connection.close()

    return redirect('/')


@app.route('/graph')
def graph():
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM parsed_responses")
    parsed_entries = cursor.fetchall()
    cursor.close()
    connection.close()

    # Create the graph
    G = nx.Graph()

    # Add nodes and edges
    for entry in parsed_entries:
        role, purpose, data_attribute, response_id = entry[0],entry[1],entry[2],entry[3]
        G.add_node(role, type='role')
        G.add_node(purpose, type='purpose')
        G.add_node(data_attribute, type='data_attribute')
        G.add_edges_from([(role, purpose), (purpose, data_attribute)])

    # Draw the graph
    plt.figure(figsize=(10, 7))
    pos = nx.spring_layout(G)
    
    random.seed(42)
    np.random.seed(42)


    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=10, font_weight='bold', edge_color='gray')

    # Save the graph to a file
    graph_path = 'static/graph.png'
    plt.savefig(graph_path)
    plt.close()

    return render_template('graph.html', graph_image=graph_path)



if __name__ == '__main__':
    app.run(debug=True)










#from flask import Flask, request, render_template_string
#from transformers import pipeline
#import mysql.connector

# Initialize the text generation pipeline with a pretrained GPT-2 model
#generator = pipeline('text-generation', model='gpt2')
#qa_pipeline = pipeline('question-answering',model='gpt2') #no model


#app = Flask(__name__)

# Configure MySQL Database
#db_config = {
#    'database': 'chatgpt_responses',
 #   'host': 'localhost',
 #   'user': 'root'  
#}

#@app.route('/')
#def index():
#    return render_template_string(open('index.html').read())


#@app.route('/process_input', methods=['POST'])
#def process_input():
#    user_input = request.form['user_input']

#    prompt = (
 #       "Role: The categories within and beyond the organization that are granted access to private information. (ie Manager, Deliverer, Analyzer, Marketer, User, etc.) People who need the information. \n"
  #      "Purpose: reason data is needed \n"
#        "Data attribute: pieces of sensitive information. Data item is an instance of a data attribute. (Attribute: age. Item: 26).\n"
 #       "This is a privacy policy tuple: Privacy Policy::=(role, purpose, data attribute) Remember these definitions. \n"
  #      "Convert this natural language into the privacy policy tuples." 
	#"An example output can look like the following: (Company, to create an account, name) \n"
        #"(Analyzer, to fight spam/malware, IP address)"
  #  )


#    prompt = (
#"Convert the following natural language into the privacy policy tuples (role, purpose, data attribute):\n\n"
#    f"Natural Language: \"{user_input}\"\n"
#    "Privacy Policy Tuples: "
#)

    # Generate text using the GPT-2 model
#    generated_text = generator(prompt, max_length=400, num_return_sequences=1)[0]['generated_text'].strip()


# Define the question and context
#    question = "How can we convert the privacy policy into a list of privacy policy tuples that follow the format of (role, purpose, data attribute) using the defintions?"
	
#    context = (
#        "Role: The categories within and beyond the organization that are granted access to private information. (ie Manager, Deliverer, Analyzer, Marketer, User, etc.) People who need the information. \n"
#        "Purpose: reason data is needed \n"
#        "Data attribute: pieces of sensitive information. Data item is an instance of a data attribute. (Attribute: age. Item: 26).\n"
#        "Privacy Policy::=(role, purpose, data attribute) Remember these definitions. \n"
#        "An example output can look like the following: (Company, to create an account, name) \n"
#        "(Analyzer, to fight spam/malware, IP address) Here is a privacy policy: \n"
#    )
    

    # Generate answer using the question answering model
#    result = qa_pipeline(question=question, context=context+user_input, max_length=1000)

    # Extract the answer from the result
#    chatgpt_response = result['answer']


    # Store the response in MySQL
#    connection = mysql.connector.connect(**db_config)
#    cursor = connection.cursor()
#    cursor.execute(
#        "INSERT INTO responses (user_input, chatgpt_response) VALUES (%s, %s)",
#        (user_input, chatgpt_response)
#    )
#    connection.commit()
#    cursor.close()
#    connection.close()

#    return f"Stored response: {chatgpt_response}"

#if __name__ == '__main__':
#    app.run(debug=True)

