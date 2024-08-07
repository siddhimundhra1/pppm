from flask import Flask, request, render_template_string, render_template, redirect, url_for, abort, jsonify
from openai import OpenAI
import mysql.connector
import re
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import numpy as np
import textwrap
import random
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from fuzzywuzzy import fuzz
from transformers import pipeline
import requests
import nltk
import spacy
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nlp = spacy.load("en_core_web_sm")
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline






# Initialize OpenAI client with your API key

client = OpenAI(api_key='')
app = Flask(__name__)

# Configure MySQL Database
db_config = {
    'database': 'chatgpt_responses',
    'host': 'localhost',
    'user': 'root'  
}


role_prompt = (
    "Definition: Role - Those granted access to private information (e.g., Manager, Deliverer, Analyzer, Marketer). People who need the information.\n"
    "Extract the role from the following natural language privacy policy sentence. If no role is mentioned, leave it out.\n\n"
    "Examples:\n"
    "Natural Language Sentence: \"In our organization, only employees who need information to perform a specific job (for example, shipping, sending gift, or analyzing) are granted access to personal information.\"\n"
    "Role: Employees\n\n"
    "Natural Language Sentence: \"Deliverers ship orders that you place.\"\n"
    "Role: Deliverers\n\n"
    "Natural Language Sentence: \"A marketer is an employee with a valid contract term, and they work under Analyzers’ supervision; the manager supervises analyzers and deliverers.\"\n"
    "Role: Marketer\n\n"
    "Natural Language Sentence: \"Our marketers will also send you a gift on your birthday.\"\n"
    "Role: Marketers\n\n"
    "Natural Language Sentence: \"We use an outside shipping company to ship orders, and a credit card processing company to bill users for goods and services.\"\n"
    "Role: We, Outside shipping company, Credit card processing company\n\n"
    "Extract all the roles in the following privacy policy. Go sentence by sentence and put in a list. Repetition is okay:\n\n"
)

purpose_prompt = (
    "Definition: Purpose - The reason or intent for which the data is collected or used. Generally a verb indicates a purpose.\n"
    "Extract the purpose from the following natural language privacy policy sentence. If no purpose is mentioned, leave it out. No commas within the purpose allowed.\n\n"
    "Examples:\n"
    "Natural Language Sentence: \"We will use your information to respond to you, regarding your purchases.\"\n"
    "Purpose: to respond to you regarding your purchases\n\n"
    "Natural Language Sentence: \"To ship your orders, deliverers access your name, order list, credit card information, address, and email address.\"\n"
    "Purpose: to ship your orders\n\n"
    "Natural Language Sentence: \"Our analyzers perform analyses on your shopping history and date of birth to enhance our services.\"\n"
    "Purpose: to enhance our services\n\n"
    "Natural Language Sentence: \"Marketers will then suggest products that might interest you.\"\n"
    "Purpose: to suggest products that might interest you\n\n"
    "Natural Language Sentence: \"Marketing staff members will send you advertisements within business hours.\"\n"
    "Purpose: to send you advertisements within business hours\n\n"
    "Natural Language Sentence: \"To send birthday gifts, marketers check your date of birth, identify you, and send a gift to your address.\"\n"
    "Purpose: to send birthday gifts\n\n"
    "Extract all the purposes in the following privacy policy. Go sentence by sentence and put in a list. Repetition is okay:\n\n"
)



data_attributes_prompt = (
    "Definition: Data Attributes - Specific pieces of information that are collected or used.\n"
    "Extract the data attributes from the following natural language privacy policy sentence. List multiple attributes within square brackets separated by commas. If no data attributes are mentioned, leave it out.\n\n"
    "Examples:\n"
    "Natural Language Sentence: \"We will use your information to respond to you, regarding your purchases.\"\n"
    "Data Attributes: [information]\n\n"
    "Natural Language Sentence: \"To ship your orders, deliverers access your name, order list, credit card information, address, and email address.\"\n"
    "Data Attributes: [name, order list, credit card information, address, email address]\n\n"
    "Natural Language Sentence: \"During registration, you are required to provide certain personal information (such as name, email, and address).\"\n"
    "Data Attributes: [name, email, address]\n\n"
    "Natural Language Sentence: \"Our analyzers perform analyses on your shopping history and date of birth.\"\n"
    "Data Attributes: [shopping history, date of birth]\n\n"
    "Natural Language Sentence: \"Marketing staff members will send you advertisements within business hours.\"\n"
    "Data Attributes: [advertisements]\n\n"
    "Natural Language Sentence: \"To send birthday gifts, marketers check your date of birth, identify you, and send a gift to your address.\"\n"
    "Data Attributes: [date of birth, address, gift]\n\n"
    "Extract all the data attributes in the following privacy policy. Go sentence by sentence and put in a list. Repetition is okay:\n\n"
)

category_prompt = (
    "Definition: Category - The general classification or grouping of the data attributes. \n"
    "If multiple data attributes (name, age, address, etc.) are clustered, a category is created. When a type of data is mentioned to be used for a purpose, a category is created. (ie. If personal information is used for shipping, personal information is a category.) \n"
    "Extract the category from the following natural language privacy policy sentence. If no category is mentioned, leave it out.\n\n"
    "Examples:\n"
    "Natural Language Sentence: \"We will use your information to respond to you, regarding your purchases.\"\n"
    "Category: User\n\n"
    "Natural Language Sentence: \"To ship your orders, deliverers access your name, order list, credit card information, address, and email address.\"\n"
    "Category: User\n\n"
    "Natural Language Sentence: \"During registration, you are required to provide certain personal information (such as name, email, and address).\"\n"
    "Category: Personal Information\n\n"
    "Natural Language Sentence: \"Our analyzers perform analyses on your shopping history and date of birth to enhance our services.\"\n"
    "Category: User\n\n"
    "Natural Language Sentence: \"Marketing staff members will send you advertisements within business hours.\"\n"
    "Category: User\n\n"
    "Natural Language Sentence: \"To send birthday gifts, marketers check your date of birth, identify you, and send a gift to your address.\"\n"
    "Category: User\n\n"
    "Now, apply this format to the following policy sentences sentence-by sentence. Repetition is okay. Put in a list:\n\n"
)


match_prompt = (
    "The above are the extracted elements. Now, match the extracted elements into the final tuple format. Use all the provided elements to form the tuples. Ensure all extracted elements are included. Ensure there are no commas within Role or Purpose. Each tuple can have multiple attributes. In addition, some tuples may be missing a role, purpose, or attribute.\n\n"
    "Format:\n"
    "(Role: <role>, Purpose: <Purpose>, Data Attributes: [<Attr1>, <Attr2>])\n\n"
    "Examples:\n"
    "Natural Language Sentence: \"The manager needs the user's email address for account verification.\"\n"
    "Matched Tuple: (Role: Manager, Purpose: for account verification, Data Attributes: [email address])\n\n"
    "Natural Language Sentence: \"We collect age and gender data from users to improve our services.\"\n"
    "Matched Tuple: (Role: , Purpose: to improve our services, Data Attributes: [age, gender])\n"
    "Natural Language Sentence: \"Deliverers require the customer's address and email to ship orders.\"\n"
    "Matched Tuple: (Role: Deliverers, Purpose: to ship orders, Data Attributes: [address, email])\n"
    "Here are the natural language privacy policy sentences. Go sentence-by-sentence using the extracted elements and put in the final tuple format.:\n"
)








def get_chatgpt_response(prompt, user_input):
    try:
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You extract specific elements from the given privacy policy sentence based on the provided definition."},
                {"role": "user", "content": prompt + user_input}
            ]
        )
      
        return completion.choices[0].message.content.strip() 
    
    except Exception as e:
        print(f"Error: {e}")
        return None
    

def match_prompt_gpt(prompt, user_input, elements):
    try:
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You extract specific elements from the given privacy policy sentence based on the provided definition."},
                {"role": "user", "content": elements+ prompt + user_input}
            ]
        )
      
        return completion.choices[0].message.content.strip() 
    
    except Exception as e:
        print(f"Error: {e}")
        return None


@app.route('/process_input', methods=['POST'])
def process_input():
    user_input = request.form['user_input']
    
    # Extract each component
    roles = get_chatgpt_response(role_prompt, user_input)
    print (roles)
    purposes = get_chatgpt_response(purpose_prompt, user_input)
    print (purposes)
    data_attributes = get_chatgpt_response(data_attributes_prompt, user_input)
    print (data_attributes)
    #categories = get_chatgpt_response(category_prompt, user_input)
    #print (categories)
    
    # Prepare the elements for matching
    extracted_elements = []
    if roles:
        extracted_elements.append(f"Roles: {roles}")
    if purposes:
        extracted_elements.append(f"Purposes: {purposes}")
    if data_attributes:
        extracted_elements.append(f"Data Attributes: {data_attributes}")
   #if categories:
   #     extracted_elements.append(f"Categories: {categories}")
    
    extracted_elements_text = "\n".join(extracted_elements)
    
    # Match the extracted elements into the final tuple
    match_input = match_prompt + extracted_elements_text
    final_tuple = match_prompt_gpt(match_prompt, user_input, extracted_elements_text)
    
    # Save the response to the database
    try:
        with mysql.connector.connect(**db_config) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO responses (user_input, chatgpt_response) VALUES (%s, %s)",
                    (user_input, final_tuple)
                )
                connection.commit()
    except Exception as e:
        print(f"Error: {e}")
        return "An error occurred while processing your request.", 500

    return redirect('/')




#HOME SCREEN

@app.route('/')
def index():
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM responses")
    entries = cursor.fetchall()
    cursor.execute("SELECT * FROM parsed_responses")
    parsed_entries = cursor.fetchall()
    cursor.execute("SELECT * FROM role_hierarchies")
    role_entries = cursor.fetchall()
    cursor.execute("SELECT * FROM superior")
    superior = cursor.fetchall()
    cursor.execute("SELECT * FROM categories")
    categories = cursor.fetchall()
    cursor.close()
    connection.close()

    return render_template_string(open('index.html').read(), entries=entries, parsed_entries=parsed_entries, role_entries=role_entries, superior=superior, categories=categories)
    
 


#PRIVACY POLICY INPUT FORM





#NEXT TWO ARE FORM ACTIONS


#DELETE BUTTON


#NEXT TWO ARE FORM ACTIONS


#DELETE BUTTON

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






#PARSE TEXT HELPER FUNCTION
#PARSE TEXT HELPER FUNCTION

def parse_text(text):
    results = []
    # Remove leading and trailing whitespace and split by ')'
    segments = text.strip().split(')')

    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
        
        # Remove leading '('
        segment = segment[1:].strip()
        
        # Extract Data Attributes separately if present
        data_attributes = ""
        if "Data Attributes:" in segment:
            before_data_attrs, after_data_attrs = segment.split("Data Attributes:", 1)
            data_attributes_part, remaining = after_data_attrs.split(']', 1)
            data_attributes = data_attributes_part.strip().strip('[]')
            segment = before_data_attrs.strip() + remaining.strip()
        
        # Split the remaining part of the segment by commas
        fields = segment.split(',')
        
        entry = {
            "Role": "",
            "Purpose": "",
            "Data Attributes": data_attributes,  # Use the extracted data attributes
            "Category": ""
        }

        for field in fields:
            if ':' in field:
                key, value = field.split(':', 1)
                key = key.strip()
                value = value.strip()
                if key == 'Role':
                    entry['Role'] = value
                elif key == 'Purpose':
                    entry['Purpose'] = value
                elif key == 'Categories':
                    entry['Category'] = value

        # Convert entry to tuple in the required format
        result_tuple = (
            entry['Role'],
            entry['Purpose'],
            entry['Data Attributes'],
            entry['Category']
        )
        results.append(result_tuple)

    return results



#PARSE TEXT BUTTON

@app.route('/parse/<int:entry_id>', methods=['POST'])
def parse_entry(entry_id):
    try:
        # Fetch the response from the responses table
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()
        query = "SELECT chatgpt_response FROM responses WHERE id = %s"
        cursor.execute(query, (entry_id,))
        response = cursor.fetchone()
        
        if response is None:
            cursor.close()
            connection.close()
            abort(404, description="Entry not found")

        response = response[0]
        cursor.close()

       
       
        

        matches = parse_text(response)
        

        

        
        parsed_entries = []

 

        for match in matches: 
            role=match[0]
            if match[0]=='':
                role="Unknown Role"
            purpose=match[1]
            if match[1]=='':
                purpose="Unknown Purpose"
            data_attributes_list=match[2]
            
            category=match[3]
            
            data_attributes_list = [attr.strip() for attr in data_attributes_list.split(',') if attr]

            for data_attribute in data_attributes_list:
                parsed_entries.append((role, purpose, data_attribute, category))
            if len(data_attributes_list)==0:
                parsed_entries.append((role, purpose, "Unknown Data Attribute", category))
        
       
       
        # Reconnect and insert data
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()
        for role, purpose, data_attribute, category in parsed_entries:
            cursor.execute(
                "INSERT INTO parsed_responses (role, purpose, data_attribute, category, response_id) VALUES (%s, %s, %s, %s, %s)",
                (role, purpose, data_attribute, category, entry_id)
            )
        connection.commit()
        cursor.close()
        connection.close()

    except mysql.connector.Error as err:
        print(f"Error: {err}")
        abort(500, description="Database error")
    
    return redirect('/')




#GRAPHING BUTTON BEGIN
#GRAPHING BUTTON BEGIN

@app.route('/graph', methods=['GET', 'POST'])
def graph():
    response_id = request.args.get('response_id')
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor()
    
    # Fetch parsed entries
    
    # Fetch parsed entries
    cursor.execute("SELECT * FROM parsed_responses WHERE response_id = %s", (response_id,))
    parsed_entries = cursor.fetchall()

    # Fetch superior-inferior relationships
    cursor.execute("SELECT superior_role, inferior_role FROM superior WHERE response_id = %s", (response_id,))
    superior_relationships = cursor.fetchall()
    

    # Fetch superior-inferior relationships
    cursor.execute("SELECT superior_role, inferior_role FROM superior WHERE response_id = %s", (response_id,))
    superior_relationships = cursor.fetchall()
    
    cursor.close()
    connection.close()

    # Create a directed graph
    G = nx.DiGraph()  # Use DiGraph for directed edges
    categories = {} 
    role_nodes = set()
    purpose_nodes = set()
    data_attribute_nodes = set()
    
   
    
    # Add nodes and edges based on parsed entries
    k = 0
   
    for entry in parsed_entries:
        role = entry[1].strip()
        purpose = entry[2].strip()
        data_attribute = entry[3].strip() + " " + str(k)
        data_attribute = entry[3].strip() + " " + str(k)
        category = entry[4].strip()
        k += 1
        
        k += 1
        
        G.add_node(role, type='role')
        G.add_node(purpose, type='purpose')
        G.add_node(data_attribute, type='data_attribute')

        if category not in categories:
            categories[category] = set()
        
        categories[category].add(data_attribute)
        role_nodes.add(role)
        purpose_nodes.add(purpose)
        data_attribute_nodes.add(data_attribute)
    
    for superior_role, inferior_role in superior_relationships:
        if superior_role not in role_nodes:
            role_nodes.add(superior_role)
        if inferior_role not in role_nodes:
            role_nodes.add(inferior_role)

    # Add directed edges based on superior-inferior relationships
    for superior_role, inferior_role in superior_relationships:
        if superior_role in role_nodes and inferior_role in role_nodes:
            G.add_edge(superior_role.strip(), inferior_role.strip())

    # Define positions
    
    for superior_role, inferior_role in superior_relationships:
        if superior_role not in role_nodes:
            role_nodes.add(superior_role)
        if inferior_role not in role_nodes:
            role_nodes.add(inferior_role)

    # Add directed edges based on superior-inferior relationships
    for superior_role, inferior_role in superior_relationships:
        if superior_role in role_nodes and inferior_role in role_nodes:
            G.add_edge(superior_role.strip(), inferior_role.strip())

    # Define positions
    pos = {}
   

   
    j = 0
    for i, node in enumerate(role_nodes):
        pos[node] = (i*2, 2+random.uniform(0,1))  # Top line for roles
       
   
       
    for i, node in enumerate(purpose_nodes):
        pos[node] = (2*i, 1)  # Middle line for purposes

    for category in categories:
        for i, node in enumerate(categories[category]):
            pos[node] = (1.5*(i + j), 0)  # Bottom line for data attributes
        j += len(categories[category])
        

    plt.figure(figsize=(20, 10))
    
    plt.figure(figsize=(20, 10))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='purple', node_size=500, alpha=0.8)
    nx.draw_networkx_nodes(G, pos, node_color='purple', node_size=500, alpha=0.8)
    
    # Draw edges with arrows
    nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=20, edge_color='black')
    # Draw edges with arrows
    nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=20, edge_color='black')
    
    # Draw labels
    def wrap_text(text, width=6):
        return '\n'.join(textwrap.wrap(text, width=width))

    wrapped_labels = {node: wrap_text(node) for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=wrapped_labels, font_size=6)
    nx.draw_networkx_labels(G, pos, labels=wrapped_labels, font_size=6)

    ax = plt.gca()
    for category, nodes in categories.items():
        x_coords = [pos[node][0] for node in nodes]
        y_coords = [pos[node][1] for node in nodes]

        if x_coords and y_coords:  # Make sure there are nodes to avoid empty boxes
            x_min, x_max = min(x_coords) - 0.5, max(x_coords) + 0.5
            y_min, y_max = min(y_coords) - 0.5, max(y_coords) + 0.5

            rect = patches.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=2,
                edgecolor='pink',
                facecolor='none',
                linestyle='--'
            )
            ax.add_patch(rect)
            label_x = (x_min + x_max) / 2
            label_y = (y_max)
            ax.text(
                label_x, label_y,
                category,
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=8,
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')
            )

    plt.xlim(min(pos.values(), key=lambda x: x[0])[0] - 1, max(pos.values(), key=lambda x: x[0])[0] + 1)
    plt.ylim(min(pos.values(), key=lambda x: x[1])[1] - 1, max(pos.values(), key=lambda x: x[1])[1] + 1)
    
    # Save the graph to a file
    timestamp = int(time.time())
    graph_path = f'static/graph_{timestamp}.png'
    plt.savefig(graph_path)
    plt.close()
    
    graph_image = f'static/graph_{timestamp}.png'
    return render_template('graph.html', graph_image=graph_image, entry_id=response_id)



#END OF GRAPHING BUTTON












#GET SET OF ROLES HELPER FUNCTION
def role_set(entry_id):
    roleset = set()
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor()
    
    # Query to select roles based on the entry_id
    query = "SELECT role FROM parsed_responses WHERE id = %s"
    cursor.execute(query, (entry_id,))
    
    # Fetch all roles
    responses = cursor.fetchall()
    
    # Add roles to the roleset
    for response in responses:
        roleset.add(response[0])
    
    connection.commit()
    cursor.close()
    connection.close()

    return roleset


#GENERATE ROLE HIERARCHY BUTTON
@app.route('/rolehierarchy/<int:entry_id>', methods=['POST'])
def role_hierarchy(entry_id):
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor()
    
    # Fetch user input from the database
    query = "SELECT user_input FROM responses WHERE id = %s"
    cursor.execute(query, (entry_id,))
    response = cursor.fetchone()
    user_input = response[0] if response else ""
    
    connection.commit()
    cursor.close()
    connection.close()
    
    # Get the roleset
    roleset = role_set(entry_id)
    
    # Create the prompt
    prompt = (
    "Find any relationship between the roles given in the privacy policy. Put them in this format: (<Superior, Inferior>). If none found, return: (). \n\n"
    "Format:\n"
    "Roles: (Analyzer, Company, Marketing, Manager, Deliverer)\n"
    "Privacy Policy: The manager supervises analyzers and deliverers.\n"
    "Output: (Manager, Analyzer), (Manager, Deliverer)\n\n"
    "Roles: " + ", ".join(roleset) + "\n"
    "Privacy Policy: " + user_input
)
    
    # Call ChatGPT model to find relationships
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You extract (<Superior>, <Inferior>) role tuples."},
            {"role": "user", "content": prompt}
        ]
    )

    chatgpt_response = completion.choices[0].message.content.strip()  

    
    # Insert ChatGPT response into the database
    
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor()
    cursor.execute(
        "INSERT INTO role_hierarchies (response_id, user_input, chatgpt_response) VALUES (%s, %s, %s)",
        (entry_id, user_input, chatgpt_response)
    )
    connection.commit()
    cursor.close()
    connection.close()
    
    return redirect('/')


@app.route('/parserole/<int:entry_id>', methods=['POST'])
def parse_role(entry_id):
    # Establish connection to the database
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor()
    
    # Fetch user input from the database
    query = "SELECT chatgpt_response FROM role_hierarchies WHERE id = %s"
    cursor.execute(query, (entry_id,))
    response = cursor.fetchone()
    chatgpt_response = response[0] if response else ""

    query = "SELECT response_id FROM role_hierarchies WHERE id = %s"
    cursor.execute(query, (entry_id,))
    response_id = cursor.fetchone()
    response_id=response_id[0]
    

    # Regex to extract role relationships
    pattern = r'\(([^,]+),\s*([^)]+)\)'
    matches = re.findall(pattern, chatgpt_response)
    role_hierarchies = [(match[0].strip(), match[1].strip()) for match in matches]

    # Insert extracted roles into the superior table
    for entry in role_hierarchies:
        cursor.execute(
            "INSERT INTO superior (response_id, superior_role, inferior_role) VALUES (%s, %s, %s)",
            (response_id, entry[0], entry[1])
        )

    connection.commit()
    cursor.close()
    connection.close()
    
    return redirect('/')


model_name = 'gpt2-large'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

def similarity_checker(text1, text2):
    prompt = f"Answer with 'Yes' or 'No'. Are '{text1}' and '{text2}' referring to the same entity?"
    inputs = tokenizer(prompt, return_tensors='pt')
    # Generate response from the model
    try:
        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=20,
            num_return_sequences=1,
            do_sample=True,        # Enable sampling
            temperature=0.7,      # Controls randomness
            top_p=0.9,            # Controls diversity
            pad_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        
        # Print the response for debugging
        print("Model Response:", response)
        
        # Interpret the response
        if 'yes' in response.lower():
            return 'Yes'
        elif 'no' in response.lower():
            return 'No'
        else:
            return 'Unable to determine'
    except Exception as e:
        print(f"Error generating text: {e}")
        return 'Error'

def find_similar_nodes(nodes):
    similar_nodes = []
    node_list = list(nodes)
    for i in range(len(node_list)):
        for j in range(i + 1, len(node_list)):
            response = similarity_checker(node_list[i], node_list[j])
            if response == 'Yes':
                similar_nodes.append((node_list[i], node_list[j]))
    return similar_nodes


"""@app.route('/flagged_nodes', methods=['GET'])
def flagged_nodes():
    entry_id = request.args.get('entry_id')
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor()
    
    # Fetch all roles
    cursor.execute("SELECT role FROM parsed_responses WHERE response_id = %s", (entry_id,))
    nodes = cursor.fetchall()
    roles_nodes = [node[0] for node in nodes]

    # Query for superior nodes
    cursor.execute("SELECT superior_role FROM superior WHERE response_id = %s", (entry_id,))
    superior_nodes = cursor.fetchall()
    superior_nodes = [node[0] for node in superior_nodes]

    # Query for inferior nodes
    cursor.execute("SELECT inferior_role FROM superior WHERE response_id = %s", (entry_id,))
    inferior_nodes = cursor.fetchall()
    inferior_nodes = [node[0] for node in inferior_nodes]

    all_nodes = set(roles_nodes + superior_nodes + inferior_nodes)
    # Find similar nodes
    similar_nodes_roles = find_similar_nodes(all_nodes)

    cursor.execute("SELECT data_attribute FROM parsed_responses WHERE response_id = %s", (entry_id,))
    nodes = cursor.fetchall()
    attributes_nodes = [node[0] for node in nodes]
    similar_nodes_attr = find_similar_nodes(attributes_nodes)
    
    cursor.close()
    connection.close()

    confirmed_pairs = [f"{pair[0]}|{pair[1]}" for pair in (similar_nodes_roles + similar_nodes_attr)]
    
    return jsonify(confirmed_pairs)"""




"""def find_prepositions_and_following_words(text):
    words = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(words)
    preposition_phrases = []
    for i, (word, pos) in enumerate(pos_tags):
        #if pos == 'IN' or pos == 'TO' or pos == 'AT':  # Prepositions have the POS tag 'IN'
            if i + 1 < len(pos_tags):
                next_word, next_pos = pos_tags[i + 1]
                if next_pos.startswith('VB') or next_pos.startswith('NN'):  # Verbs start with 'VB', nouns with 'NN'
                    preposition_phrases.append(f"{word} {next_word}")
    return preposition_phrases"""

model_name = "bert-large-uncased"  # Choose an appropriate model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Initialize Transformers pipeline
classifier = pipeline('text-classification', model=model, tokenizer=tokenizer)


def find_purpose_phrases(text):
    # Tokenize and POS tag using NLTK
    words = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(words)
    
    # Process text with SpaCy
    doc = nlp(text)
    
    purpose_phrases = []
    
    # Iterate over tokens to find prepositions and related phrases
    i = 0
    while i < len(pos_tags):
        word, pos = pos_tags[i]
        if pos == 'IN' or pos == 'TO' or pos == 'AT':  # Look for the preposition
            phrase = [word]  # Start with the preposition
            i += 1
            
            # Use SpaCy dependency parsing to get the head and children
            while i < len(pos_tags):
                next_word, next_pos = pos_tags[i]
                
                # Check if the next word is a verb or noun
                if next_pos.startswith('VB') or next_pos.startswith('NN') or next_pos.startswith('DT'):
                    phrase.append(next_word)
                else:
                    break  # Stop at non-verb or non-noun
                
                i += 1
            
            if len(phrase) > 1:  # Ensure it's more than just a single preposition
                purpose_phrases.append(' '.join(phrase))
        else:
            i += 1
    
    # Use SpaCy to enhance purpose extraction
    for token in doc:
        if token.dep_ == 'prep' and token.head.pos_ == 'VERB':
            phrase = [token.head.text, token.text]
            for child in token.children:
                if child.dep_ in ['pobj', 'dobj']:
                    phrase.append(child.text)
            if len(phrase) > 1:
                purpose_phrases.append(' '.join(phrase))
    
    return purpose_phrases

def find_noun_phrases(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    noun_phrases = [chunk.text for chunk in doc.noun_chunks]
    return noun_phrases

def find_similar_nodes(nodes):
    # Placeholder function for finding similar nodes
    # Replace this with actual logic to find similar nodes
    return [(node, node) for node in nodes]

@app.route('/flagged_nodes', methods=['GET'])
def flagged_nodes():
    entry_id = request.args.get('entry_id')
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor()

    # Fetch chatgpt_response
    cursor.execute("SELECT user_input FROM responses WHERE id = %s", (entry_id,))
    result = cursor.fetchone()
    if result:
        chatgpt_response = result[0]
    else:
        chatgpt_response = ""

    cursor.close()
    connection.close()

    # Find preposition+verb or preposition+noun in chatgpt_response
    prep_phrases = find_purpose_phrases(chatgpt_response)
    noun_phrases=find_noun_phrases(chatgpt_response)

    # Print the preposition phrases
    #print("\nPreposition + Verb or Noun in chatgpt_response:")
    #for phrase in prep_phrases:
     #   print(phrase)
    print (noun_phrases)
    print (prep_phrases)
    
    # Convert list to a string with newlines for return
    response_content = "\n".join(prep_phrases)
    return response_content, 200


#WORKING ON THIS http://127.0.0.1:5000

@app.route('/update_graph', methods=['POST'])
def update_graph():
    confirmed_pairs = request.form.getlist('confirmed_pairs')
    entry_id = request.form.get('entry_id')
    
    # Fetch data again to maintain consistency
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor()
    
    # Fetch parsed entries
    cursor.execute("SELECT * FROM parsed_responses WHERE response_id = %s", (entry_id,))
    parsed_entries = cursor.fetchall()

    # Fetch superior-inferior relationships
    cursor.execute("SELECT superior_role, inferior_role FROM superior WHERE response_id = %s", (entry_id,))
    superior_relationships = cursor.fetchall()
    
    cursor.close()
    connection.close()
    
    # Update relationships in-memory
    mapping = {}
    for pair in confirmed_pairs:
        node1, node2 = pair.split('|')
        node1=node1.strip()
        node2=node2.strip()
        mapping[node2] = node1
    print (mapping)

    response_id = request.args.get('response_id')
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor()
    

    # Create a directed graph
    G = nx.DiGraph()  # Use DiGraph for directed edges
    categories = {} 
    role_nodes = set()
    purpose_nodes = set()
    data_attribute_nodes = set()
    
    # Add nodes and edges based on parsed entries
    k = 0
    
    for entry in parsed_entries:
        if entry[1].strip() not in mapping.keys(): 
            role = entry[1].strip() 
            G.add_node(role, type='role')
            role_nodes.add(role)
       

        
       
            
       
            
        purpose = entry[2].strip()    
        category = entry[4].strip()
        k += 1   
        G.add_node(purpose, type='purpose')
        

        if category not in categories:
            categories[category] = set()

        if entry[3].strip() not in mapping.keys():
            data_attribute = entry[3].strip() + " " + str(k)
            G.add_node(data_attribute, type='data_attribute')
            data_attribute_nodes.add(data_attribute)
        if entry[3].strip() not in mapping.keys():
            categories[category].add(data_attribute) 
        else:
            categories[category].add(mapping[entry[3].strip()])
       
        purpose_nodes.add(purpose)
        
    for entry in superior_relationships:
        if entry[1].strip() not in mapping.keys(): 
            role = entry[1].strip() 
            G.add_node(role, type='role')
            role_nodes.add(role)
        else:
            role = mapping[entry[1].strip()] 
            G.add_node(role, type='role')
            role_nodes.add(role)
    
    # Add directed edges based on superior-inferior relationships
    for superior_role, inferior_role in superior_relationships:
        superior_role=superior_role.strip()
        inferior_role=inferior_role.strip()
        if superior_role in role_nodes and inferior_role in role_nodes:
            G.add_edge(superior_role, inferior_role)
        elif superior_role in mapping.keys() and inferior_role in role_nodes:
            G.add_edge(mapping[superior_role], inferior_role)
        elif superior_role in role_nodes and inferior_role in mapping.keys():
            G.add_edge(superior_role, mapping[inferior_role])
        elif superior_role in mapping.keys() and inferior_role in mapping.keys():
            G.add_edge(mapping[superior_role], mapping[inferior_role])
    print (superior_relationships)
    print (role_nodes)
        

    # Define positions
    pos = {}
   
    j = 0
    for i, node in enumerate(role_nodes):
        pos[node] = (i*2, 2+random.uniform(0,1))  # Top line for roles
       
    for i, node in enumerate(purpose_nodes):
        pos[node] = (2*i, 1)  # Middle line for purposes
    for category in categories: #FIX
        for i, node in enumerate(categories[category]):
            pos[node] = (1.5*(i + j), 0)  # Bottom line for data attributes
        j += len(categories[category])

    plt.figure(figsize=(20, 10))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='purple', node_size=500, alpha=0.8)
    
    # Draw edges with arrows
    nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=20, edge_color='black')
    
    # Draw labels
    def wrap_text(text, width=6):
        return '\n'.join(textwrap.wrap(text, width=width))

    wrapped_labels = {node: wrap_text(node) for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=wrapped_labels, font_size=6)

    ax = plt.gca()
    for category, nodes in categories.items():
        x_coords = [pos[node][0] for node in nodes]
        y_coords = [pos[node][1] for node in nodes]

        if x_coords and y_coords:  # Make sure there are nodes to avoid empty boxes
            x_min, x_max = min(x_coords) - 0.5, max(x_coords) + 0.5
            y_min, y_max = min(y_coords) - 0.5, max(y_coords) + 0.5

            rect = patches.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=2,
                edgecolor='pink',
                facecolor='none',
                linestyle='--'
            )
            ax.add_patch(rect)
            label_x = (x_min + x_max) / 2
            label_y = (y_max)
            ax.text(
                label_x, label_y,
                category,
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=8,
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')
            )

    plt.xlim(min(pos.values(), key=lambda x: x[0])[0] - 1, max(pos.values(), key=lambda x: x[0])[0] + 1)
    plt.ylim(min(pos.values(), key=lambda x: x[1])[1] - 1, max(pos.values(), key=lambda x: x[1])[1] + 1)
    
    # Save the graph to a file
    timestamp = int(time.time())
    graph_path = f'static/graph_{timestamp}.png'
    timestamp = int(time.time())
    graph_path = f'static/graph_{timestamp}.png'
    plt.savefig(graph_path)
    plt.close()
    
    graph_image = f'static/graph_{timestamp}.png'
    return render_template('graph.html', graph_image=graph_image, entry_id=response_id)
    
    




@app.route('/categories/<int:entry_id>', methods=['POST'])
def category_finder(entry_id):
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor()
    
    # Fetch user input from the database
    query = "SELECT user_input FROM responses WHERE id = %s"
    cursor.execute(query, (entry_id,))
    response = cursor.fetchone()
    user_input = response[0] if response else ""
    
    connection.commit()
    cursor.close()
    connection.close()
     
    
    # Create the prompt
    prompt = (
    "Definition - Data Attribute: Pieces of sensitive information. A data item is an instance of a data attribute (e.g., Attribute: age; Item: 26).\n"
    "Definition - Category: categories of data collected by a service, each containing specific attributes that represent different types of information\n\n" 
    "Find all the categories in the given privacy policy and list them in the format [Category1, Category2, ...].\n"
    "Privacy Policy: " + user_input
)

    
    # Call ChatGPT model to find relationships
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You extract a list of categories."},
            {"role": "user", "content": prompt}
        ]
    )

    chatgpt_response = completion.choices[0].message.content.strip()  

    
    # Insert ChatGPT response into the database
    
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor()
    cursor.execute(
        "INSERT INTO categories (response_id, user_input, chatgpt_response) VALUES (%s, %s, %s)",
        (entry_id, user_input, chatgpt_response)
    )
    connection.commit()
    cursor.close()
    connection.close()
    
    return redirect('/')











if __name__ == '__main__':
    app.run(debug=True)








"""   Example:\n"
    "Privacy Policy: 'The manager needs the user's email address for account verification.'\n"
    "Categories: [User]\n"
    "Privacy Policy: 'The marketing team uses customer purchase history for targeted advertising.'\n"
    "Categories: [Customer]\n"
    "Privacy Policy: 'We collect age and gender data from users to improve our services.'\n"
    "Categories: [User]\n"
    "Privacy Policy: 'Employees have access to contact information for internal communications.'\n"
    "Categories: []\n"
    "Privacy Policy: 'Child's name and school are collected for registration purposes.'\n"
    "Categories: [Child]\n




    Groups of attributes. They describe what kind of data the attributes describe. (e.g., User, Device, Child).
"""





"""
@app.route('/process_input', methods=['POST'])
def process_input():
    user_input = request.form['user_input']
    
   


    prompt = (
    "Definition - Role: Those granted access to private information. (ie Manager, Deliverer, Analyzer, Marketer, etc.) People who need the information. \n"
    "I want you to extract the role, purpose, data attribute, and category from the following natural language privacy policy sentence."
    "Use the format provided below for the extraction. If an element is not mentioned, leave it out. No commas allowed. Response should only consist of tuples.\n\n"
    "Format:\n"
    "(Role: <Role>, Purpose: <Purpose>, [Data Attribute: <Data Attribute>, ...], Category: <Category>)\n\n"
    "Natural Language Sentence: \"The manager needs the user's email address for account verification.\"\n\n"
    "Formal Language Translation:\n"
    "(Role: Manager, Purpose: for account verification, Data Attributes: [email address], Category: User)\n\n"
    "Natural Language Sentence: \"The marketing team uses customer purchase history for targeted advertising.\"\n\n"
    "Formal Language Translation:\n"
    "(Role: Marketing team, Purpose: for targeted advertising, Data Attributes: [purchase history], Category: Customer)\n\n"
    "Natural Language Sentence: \"We collect age and gender data from users to improve our services.\"\n\n"
    "Formal Language Translation:\n"
    "(Purpose: to improve our services, Data Attributes: [age, gender], Category: User)\n\n"
    "Natural Language Sentence: \"Employees have access to contact information for internal communications.\"\n\n"
    "Formal Language Translation:\n"
    "(Role: Employees, Purpose: for internal communications, Data Attributes: [contact information])\n\n"
    "Natural Language Sentence: \"Child's name and school are collected for registration purposes.\"\n\n"
    "Formal Language Translation:\n"
    "(Purpose: for registration purposes, Data Attributes: [name, school], Category: Child)\n\n"
    "Now, apply this format to the following policy sentence by sentence:\n\n"
)



    # Process input with ChatGPT
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
           {"role": "system", "content": "You extract (role, purpose, data attribute, category) tuples from the given privacy policy. Generate tuples strictly based on words explicitly stated in the privacy policy. No numbering or commas in purposes."},
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
    return redirect('/')
    #return f"Stored response: {chatgpt_response}"
"""




"""match_prompt = (
    "The above are the extracted elements. Now, match the extracted elements into the final tuple format. Use all the provided elements to form the tuples. Ensure all extracted elements are included. Ensure there are no commas within Role or Purpose. Each tuples can have multiple attributes and categories.\n\n"
    "Format:\n"
    "(Role: <role>, Purpose: <Purpose>, Data Attributes: [<Attr1>, <Attr2>], Categories: [<Category1>, <Category2>])\n\n"
    "Examples:\n"
    "Natural Language Sentence: \"The manager needs the user's email address for account verification.\"\n"
    "Matched Tuple: (Role: Manager, Purpose: for account verification, Data Attributes: [email address], Categories: [User])\n\n"
    "Natural Language Sentence: \"We collect age and gender data from users to improve our services.\"\n"
    "Matched Tuple: (Role: , Purpose: to improve our services, Data Attributes: [age], Categories: [User])\n"
    "Matched Tuple: (Role: , Purpose: to improve our services, Data Attributes: [gender], Categories: [User])\n\n"
    "Natural Language Sentence: \"Deliverers require the customer's address and email to ship orders.\"\n"
    "Matched Tuple: (Role: Deliverers, Purpose: to ship orders, Data Attributes: [address], Categories: [Customer])\n"
    "Matched Tuple: (Role: Deliverers, Purpose: to ship orders, Data Attributes: [email], Categories: [Customer])\n\n"
    "Here are the natural language privacy policy sentences. Go sentence-by-sentence using the extracted tuples.:\n"
)"""










