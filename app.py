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


# Initialize OpenAI client with your API key
client = OpenAI(api_key='#KEY')

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
    "Definition - Role: Those granted access to private information. (ie Manager, Deliverer, Analyzer, Marketer, etc.) People who need the information. \n"
    "I want you to extract the role, purpose, data attribute, and category from the following natural language privacy policy sentence. "
    "Use the format provided below for the extraction. If an element is not mentioned, leave it out.\n\n"
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
           {"role": "system", "content": "You extract (role, purpose, data attribute) tuples from the given privacy policy. Generate tuples strictly based on words explicitly stated in the privacy policy. No numbering or commas in purposes."},
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

        # Define the regex pattern
        pattern = re.compile(
    r"\(?(?:\s*Role:\s*(?P<Role>[^,]*)\s*,\s*)?"  # Optional Role
    r"(?:\s*Purpose:\s*(?P<Purpose>.*?)(?=\s*Data Attributes:))"  # Optional Purpose
    r"\s*Data Attributes:\s*\[(?P<DataAttributes>[^\]]+)\]\s*,\s*"  # Required Data Attributes
    r"(?:\s*Category:\s*(?P<Category>[^\)]+)\s*)?"  # Optional Category
    r"\)?"
)

        
        # Extract tuples from response text
        matches = re.findall(pattern, response)
        
        parsed_entries = []
        for match in matches:
            '''role, purpose, data_attributes, category = match
    
            # Default values if fields are missing
            role = role if role else 'Unknown Role'
            purpose = purpose if purpose else 'Unknown Purpose'
            category = category if category else 'Unknown Category'

    
            # Split the data attributes by comma and strip any extra spaces
            data_attributes_list = [attr.strip() for attr in data_attributes.split(',') if attr]''' 
            role=match[0]
            purpose=match[1]
            data_attributes_list=match[2]
            category=match[3]
            data_attributes_list = [attr.strip() for attr in data_attributes_list.split(',') if attr]

            for data_attribute in data_attributes_list:
                parsed_entries.append((role, purpose, data_attribute, category))
        print (matches)
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


















'''@app.route('/parse/<int:entry_id>', methods=['POST'])
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

    return redirect('/')'''


@app.route('/graph', methods=['GET', 'POST'])
def graph():
    response_id = request.args.get('response_id')
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM parsed_responses WHERE response_id = %s", (response_id,))
    parsed_entries = cursor.fetchall()
    cursor.close()
    connection.close()

    # Create the graph
    G = nx.Graph()
    categories = {} 
    role_nodes = set()
    purpose_nodes = set()
    data_attribute_nodes = set()
    # Add nodes and edges
    k=0
    for entry in parsed_entries:
        role = entry[1].strip()
        purpose = entry[2].strip()
        data_attribute = entry[3].strip()+" "+str(k)
        category = entry[4].strip()
        k+=1
        G.add_node(role, type='role')
        G.add_node(purpose, type='purpose')
        G.add_node(data_attribute, type='data_attribute')
        
        #G.add_edge(role, purpose, style='dashed')

        if category not in categories:
            categories[category] = set()
        
        categories[category].add(data_attribute)
        role_nodes.add(role)
        purpose_nodes.add(purpose)
        data_attribute_nodes.add(data_attribute)


    

    


    #Define positions
    pos = {}
    j=0
    for i, node in enumerate(role_nodes):
        pos[node] = (i, 2)  # Top line for roles
    for i, node in enumerate(purpose_nodes):
        pos[node] = (i, 1)  # Middle line for purposes
    for category in categories:
        for i, node in enumerate(categories[category]):
            pos[node] = (i+j, 0)  # Bottom line for data attributes
        j+=len(categories[category])


    plt.figure(figsize=(14, 10))
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='blue', node_size=500, alpha=0.8)
    

    
    # Draw labels
    def wrap_text(text, width=6):
        return '\n'.join(textwrap.wrap(text, width=width))

    wrapped_labels = {node: wrap_text(node) for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=wrapped_labels, font_size=5)


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
                edgecolor='red',
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
    graph_path = 'static/graph.png'
    plt.savefig(graph_path)
    plt.close()
    print (categories)
    return render_template('graph.html', graph_image=graph_path)




if __name__ == '__main__':
    app.run(debug=True)










