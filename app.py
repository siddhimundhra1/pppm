from flask import Flask, request, render_template_string, render_template, redirect, url_for, abort
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
from fuzzywuzzy import fuzz




# Initialize OpenAI client with your API key


app = Flask(__name__)

# Configure MySQL Database
db_config = {
    'database': 'chatgpt_responses',
    'host': 'localhost',
    'user': 'root'  
}





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
    cursor.close()
    connection.close()

    return render_template_string(open('index.html').read(), entries=entries, parsed_entries=parsed_entries, role_entries=role_entries, superior=superior)
    
 


#PRIVACY POLICY INPUT FORM


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



    '''prompt = (
    "I want you to extract the role, purpose, data attribute, and category from the following natural language privacy policy sentence. "
    "Use the format provided below for the extraction. If an element is not mentioned, leave it out.\n\n"
    "Format:\n"
    "(Role: <Role>, Purpose: <Purpose>, Data Attribute: <Data Attribute>, Category: <Category>)\n\n"
    "Natural Language Sentence: \"The manager needs the user's email address for account verification.\"\n\n"
    "Formal Language Translation:\n"
    "(Role: Manager, Purpose: for account verification, Data Attribute: email address, Category: User)\n\n"
    "Natural Language Sentence: \"The marketing team uses customer purchase history for targeted advertising.\"\n\n"
    "Formal Language Translation:\n"
    "(Role: Marketing team, Purpose: for targeted advertising, Data Attribute: purchase history, Category: Customer)\n\n"
    "Natural Language Sentence: \"We collect age and gender data from users to improve our services.\"\n\n"
    "Formal Language Translation:\n"
    "(Purpose: to improve our services, Data Attribute: age, Data Attribute: gender, Category: User)\n\n"
    "Natural Language Sentence: \"Employees have access to contact information for internal communications.\"\n\n"
    "Formal Language Translation:\n"
    "(Role: Employees, Purpose: for internal communications, Data Attribute: contact information)\n\n"
    "Natural Language Sentence: \"Child's name and school are collected for registration purposes.\"\n\n"
    "Formal Language Translation:\n"
    "(Purpose: for registration purposes, Data Attribute: name, Data Attribute: school, Category: Child)\n\n"
)'''
    

    


    """prompt = (
      "Role: The categories within and beyond the organization that are granted access to private information. (ie Manager, Deliverer, Analyzer, Marketer, User, etc.) People who need the information. \n"
       "Purpose: reason data is needed \n"
       "Data attribute: pieces of sensitive information. Data item is an instance of a data attribute. (Attribute: age. Item: 26).\n"
       "Privacy Policy::=(role, purpose, data attribute) Remember these definitions. \n"
       "Convert this natural language into the privacy policy tuples. An example output can look like the following: (Company, to create an account, name) \n"
       "(Analyzer, to fight spam/malware, IP address) \n"
       "(We, to share content, gender) \n"
   )"""

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
                elif key == 'Category':
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

@app.route('/graph', methods=['GET', 'POST'])
def graph():
    response_id = request.args.get('response_id')
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor()
    
    # Fetch parsed entries
    cursor.execute("SELECT * FROM parsed_responses WHERE response_id = %s", (response_id,))
    parsed_entries = cursor.fetchall()

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
        category = entry[4].strip()
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


def find_similar_nodes(nodes, threshold=70):
    similar_nodes = []
    node_list = list(nodes)
    for i in range(len(node_list)):
        for j in range(i + 1, len(node_list)):
            similarity = fuzz.ratio(node_list[i], node_list[j])
            if similarity > threshold:
                similar_nodes.append((node_list[i], node_list[j], similarity))
    return similar_nodes

@app.route('/flagged_nodes', methods=['GET', 'POST'])
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
    similar_nodes_attr=find_similar_nodes(attributes_nodes)
    
    
    #if request.method == 'POST':
        # Handle user confirmation
     #   confirmed_pairs = request.form.getlist('confirmed_pairs')
        # Update database based on user confirmation
        # ... (e.g., merging nodes or updating records)
      #  update_graph_with_confirmations(confirmed_pairs)
       # return redirect('/')
    
    cursor.close()
    connection.close()
    
    return render_template('flagged_nodes.html', similar_nodes=similar_nodes_roles+similar_nodes_attr, entry_id=entry_id)


#WORKING ON THIS

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
    plt.savefig(graph_path)
    plt.close()
    
    graph_image = f'static/graph_{timestamp}.png'
    return render_template('graph.html', graph_image=graph_image, entry_id=response_id)
    
    






if __name__ == '__main__':
    app.run(debug=True)










