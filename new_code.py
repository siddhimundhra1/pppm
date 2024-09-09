import nltk
import spacy
import matplotlib.pyplot as plt
import networkx as nx
import random
import textwrap
import matplotlib.patches as patches
import time
import requests
import re
from collections import defaultdict
import time
import plotly.graph_objects as go

def preprocess_text(text):
    """Preprocess text to handle irregular formatting."""
    # Remove extra newlines and whitespace
    
    text = text.replace('\n', ' ').replace('\r', ' ').strip()
    return ' '.join(text.split())

def extract_node_sentences(text, G):
    nlp = spacy.load("en_core_web_sm")  # Load a pre-trained spaCy model
    text=preprocess_text(text)
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    
    # Initialize a dictionary to map nodes to sentences
    node_sentences = {}
    
    for node in G.nodes():
        node_type = G.nodes[node].get('type')
        node_key = node.split(":")[1].strip().lower() if ":" in node else node.strip().lower()
        
        if node_type:
            # Check if node's key exists in the text
            if node_key in text.strip().lower():
                # Find the sentence containing the node's key
                found_sentence = ""
                for sentence in sentences:
                    if node_key in sentence.strip().lower():
                        found_sentence = sentence
                        break
                node_sentences[node] = found_sentence
            else:
                node_sentences[node] = ""
    
    return node_sentences

def match_role_purpose(roles,purposes, text):
    matched_tuples=[]
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    roles=[item.split(":")[1].strip() for item in roles if item.split(":")[1].strip().lower() in doc.text.lower()]
    purposes=[item.split(":")[1].strip() for item in purposes if item.split(":")[1].strip().lower() in doc.text.lower()]
    for sentence in sentences:
        for role in roles:
            if role in sentence:
                for purpose in purposes: 
                    if purpose in sentence:
                        candidate_tuple=(role,purpose)
                        if candidate_tuple not in matched_tuples:
                                    matched_tuples.append(candidate_tuple)
    return matched_tuples

def match_purpose_attribute(purposes,attributes, text):
    matched_tuples=[]
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    purposes=[item.split(":")[1].strip() for item in purposes if item.split(":")[1].strip().lower() in doc.text.lower()]
    attributes=[item.split(":")[1].strip() for item in attributes if item.split(":")[1].strip().lower() in doc.text.lower()]
    for sentence in sentences:
        for purpose in purposes:
            if purpose in sentence:
                for attribute in attributes: 
                    if attribute in sentence:
                        candidate_tuple=(purpose,attribute)
                        if candidate_tuple not in matched_tuples:
                                    matched_tuples.append(candidate_tuple)
    return matched_tuples



# Load SpaCy model
nlp = spacy.load('en_core_web_sm')
def find_purpose_phrases(text):
    # Process text with SpaCy
    doc = nlp(text)
    
    purpose_phrases = []
    processed_tokens = set()  # To keep track of already processed tokens

    # Function to extract purpose phrases from the subtree of a token
    def extract_phrase_from_subtree(token):
        phrase = []
        for child in token.subtree:
            if (child.dep_ != 'punct') and (child.pos_ == 'VERB' or child.pos_ == 'NOUN' or child.pos_ == 'DET') and (child not in processed_tokens):
                phrase.append(child.text)
                processed_tokens.add(child)  # Mark token as processed
        return ' '.join(phrase).strip() #.replace(',', '') 

    # Iterate over tokens in the document
    for token in doc:
        # Check for infinitive constructions (e.g., 'to') and specific purpose-indicating words
        if (token.text.lower() == 'to' and token.dep_ == 'mark' and token.head.pos_ == 'VERB') or \
           (token.pos_ == 'VERB' and token.dep_ not in ['aux', 'auxpass']):
            # Use token.head for 'to' markers, or token directly for other verbs
            phrase = extract_phrase_from_subtree(token.head if token.text.lower() == 'to' else token)
            if phrase and phrase not in purpose_phrases:
                purpose_phrases.append(phrase)

    return purpose_phrases



def find_noun_phrases(text):
    doc = nlp(text)
    noun_phrases = []

    for chunk in doc.noun_chunks:
        # Check if the noun chunk is not a subject
        if chunk.root.dep_ not in ['nsubj', 'nsubjpass']:
            noun_phrases.append(chunk.text)
    
    return noun_phrases

def find_subject_phrases(text):
    doc = nlp(text)
    subjects = []
    for token in doc:
        if token.dep_ in ('nsubj', 'nsubjpass') and token.text.lower() != 'you':
            subjects.append(token.text)
    return subjects

# Testing
def test_functions():
    text = """
When you create an Account, use a ChatterBaby product, download a software update, contact us
or participate in an online survey or data donation, we may collect a variety of information, including
your name, age, mailing address, phone number, email address, contact preferences, credit card
information, username and password. You acknowledge that this information may be personal to you,
and by creating an Account on the Services and providing Personal Information to us, you allow others,
including us, to identify you and therefore may not be anonymous.
   When you set up ChatterBaby, you’ll be asked several questions including child’s name, e-mail,
   child’s date of birth, week of delivery and gender. This information is used to collect, measure, and
   process autism risk.
   We will process the audio recordings from the device. We will also process any video data you may
   upload, extracting acoustic features. We may also process information from the Services so that we can
   send you alerts when something happens. We will store this data indefinitely.

    """
    text=text.strip()


    







# Set your API key
    api_key = #API_KEY

# Set the request URL
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"

# Define the headers
    headers = {
        'Content-Type': 'application/json',
    }

# Define the payload
    gem_attribute = "In the context of privacy policies, an attribute refers to pieces of information like 'name' or 'age'. Given the following privacy policy text, identify and list all possible attributes mentioned word for word. Format the output as a comma-separated list enclosed in square brackets, like [attr1, attr2, attr3].\n\n" + text
    attr_payload = {
    "contents": [
        {
            "parts": [
                {"text": gem_attribute}
            ]
        }
    ],
    "safetySettings": [
    
    {
      "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
      "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE"
    }
  ]
}
    
    purpose_prompt = (
    "In the context of privacy policies, a purpose refers to the reasons for collecting and using personal data, such as 'to provide services' or 'to process transactions'. Given the following privacy policy text, identify and list all possible purposes mentioned word for word. Format the output as a semi-colon-separated list enclosed in square brackets, like [purpose1; purpose2; purpose3].\n\n" + text
)
    purpose_payload = {
    "contents": [
        {
            "parts": [
                {"text": purpose_prompt}
            ]
        }
    ],
    "safetySettings": [
       
    {
      "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
      "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE"
    }
  ]
}
    
    role_prompt = (
    "In the context of privacy policies, a role refers to the recepient of user data not including the user themself. Given the following privacy policy text, identify and list all possible roles mentioned. Format the output as a comma-separated list enclosed in square brackets, like [role1, role2, role3].\n\n" + text
)
    
    role_payload = {
    "contents": [
        {
            "parts": [
                {"text": role_prompt}
            ]
        }
    ],
    "safetySettings": [
        
    {
      "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
      "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE"
    }
  ]
}
    role_purpose_prompt = (
    "In the context of privacy policies, a role-purpose tuple represents the relationship between data recepients who get access to user data (roles) such as marketing or deliverers and their reasons for accessing or using personal data (purposes). Given the following privacy policy text, identify and list all role-purpose tuples word for word. Each tuple should be formatted as (role; purpose) and enclosed in square brackets, like [(role1; purpose1), (role2; purpose2), ...].\n\n" + text
)
    rp_payload = {
    "contents": [
        {
            "parts": [
                {"text": role_purpose_prompt}
            ]
        }
    ],
    "safetySettings": [
      
    {
      "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
      "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE"
    }
  ]
}
    purpose_attribute_prompt = (
    "In the context of privacy policies, a purpose-attribute tuple represents the relationship between the reasons the company needs personal data (purposes) and the specific pieces of personal data (attributes). Purposes are not user actions. For example, if address is needed to ship orders, then (address, to ship orders) should be output. Given the following privacy policy text, identify and list all purpose-attribute tuples making sure everything is word for word. Each tuple should be formatted as (purpose; attribute) and enclosed in square brackets, like [(purpose1; attribute1), (purpose2, attribute2), ...].\n\n" + text
)
    
    pa_payload = {
    "contents": [
        {
            "parts": [
                {"text": purpose_attribute_prompt}
            ]
        }
    ],
    "safetySettings": [
        
    {
      "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
      "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE"
    }
  ]
}







# Make the POST request

#DATA ATTRIBUTE
    generated_text=""
    for x in range(4):
        response = requests.post(url, headers=headers, json=attr_payload)
        response = response.json()
        print (response)
        generated_text += response['candidates'][0]['content']['parts'][0]['text']
    words = re.findall(r'\[(.*?)\]', generated_text)
    attribute_gen = []
    for word in words:
        attribute_gen.extend(word.strip().split(', '))
    print(attribute_gen)
    

#PURPOSES
    generated_text=""
    for x in range(4):
        response = requests.post(url, headers=headers, json=purpose_payload)
        response = response.json()
        generated_text += response['candidates'][0]['content']['parts'][0]['text']
    words = re.findall(r'\[(.*?)\]', generated_text)
    purpose_gen = []
    for word in words:
        purpose_gen.extend(word.strip().split(';'))
    print(purpose_gen)

#ROLES
    generated_text=""
    for x in range(4):
        response = requests.post(url, headers=headers, json=role_payload)
        response = response.json()

        generated_text += response['candidates'][0]['content']['parts'][0]['text']
    words = re.findall(r'\[(.*?)\]', generated_text)
    role_gen = []
    for word in words:
        role_gen.extend(word.strip().split(', '))
    print(role_gen)
    
    time.sleep(70)
#ROLE-PURPOSE
    generated_text=""
    for x in range(6):
        response = requests.post(url, headers=headers, json=rp_payload)
        response = response.json()
        generated_text += response['candidates'][0]['content']['parts'][0]['text']
    words = re.findall(r'\([^\)]+\)',generated_text)
    rp_gen = []
    for word in words:
        rp_tuple = tuple(item.strip() for item in word.strip('()').split(';'))
        rp_gen.append(rp_tuple)        
    print(rp_gen)

#PURPOSE-ATTRIBUTE
    generated_text=""
    for x in range(6):
        response = requests.post(url, headers=headers, json=pa_payload)
        response = response.json()
        print (response)
        generated_text += response['candidates'][0]['content']['parts'][0]['text']
    words = re.findall(r'\([^\)]+\)',generated_text)
    pa_gen = []
    for word in words:
        pa_tuple = tuple(item.strip() for item in word.strip('()').split(';'))
        pa_gen.append(pa_tuple)        
    print(pa_gen)








    print("\nPurposes:")
    purpose_phrases = set(find_purpose_phrases(text))
    purpose_dict={}
    for purpose in purpose_phrases:
        purpose_dict["Purpose: "+purpose.strip().lower()]=1
    
    for purpose in purpose_gen:
        purpose="Purpose: "+purpose.strip().lower()
        if purpose in purpose_dict.keys():
            purpose_dict[purpose]+=1
        else:
            purpose_dict[purpose]=1



    pa_gen_purpose=set([entry[0].strip().lower() for entry in pa_gen if len(entry)>=2])
    for purpose in pa_gen_purpose:
        purpose="Purpose: "+purpose
        if purpose in purpose_dict.keys():
                purpose_dict[purpose]+=1
        else:
            purpose_dict[purpose]=1   

    rp_gen_purpose=set([entry[1].strip().lower() for entry in rp_gen if len(entry)>=2])
    for purpose in rp_gen_purpose:
        purpose="Purpose: "+purpose
        if purpose in purpose_dict.keys():
                purpose_dict[purpose]+=1
        else:
            purpose_dict[purpose]=1     
    print(purpose_dict)


    print("\nAttribute:")
    noun_phrases = set(find_noun_phrases(text))
    attr_dict={}
    for attr in noun_phrases:
        attr_dict["Attr: "+attr.strip().lower()]=1
    
    for attr in attribute_gen:
        attr="Attr: "+attr.strip().lower()
        if attr in attr_dict.keys():
            attr_dict[attr]+=1
        else:
            attr_dict[attr]=1

    pa_gen_attr=set([entry[1].strip().lower() for entry in pa_gen if len(entry)>=2])
    for attr in pa_gen_attr:
        attr="Attr: "+attr
        if attr in attr_dict.keys():
                attr_dict[attr]+=1
        else:
            attr_dict[attr]=1

    print(attr_dict)


    print("\nRoles:")
    subject_phrases = set(find_subject_phrases(text))
    role_dict={}
    for role in subject_phrases:
        role_dict["Role: "+role.strip().lower()]=1

    for role in role_gen:
        role="Role: "+role.strip().lower()
        if role in role_dict.keys():
            role_dict[role]+=1
        else:
            role_dict[role]=1
    
    rp_gen_roles=set([entry[0].strip().lower() for entry in rp_gen if len(entry)>=2])
    for role in rp_gen_roles:
        role="Role: "+role
        if role in role_dict.keys():
                role_dict[role]+=1
        else:
            role_dict[role]=1
            
    print(role_dict)



    
   
    create_graph(role_dict,purpose_dict,attr_dict,rp_gen,pa_gen, text)







def create_graph(role_dict,purpose_dict,attr_dict,rp_gen,pa_gen,text):
    # Create a directed graph
    
    role_nodes = [item for item in role_dict.keys() if role_dict[item]>=3]
    purpose_nodes = [item for item in purpose_dict.keys() if purpose_dict[item]>=2]
    data_attribute_nodes = [item for item in attr_dict.keys() if attr_dict[item]>=3]

        




    rp_con=match_role_purpose(role_nodes,purpose_nodes,text)
    rp_dict={}
    for entry in rp_con:
        if len(entry)>=2:
            tup = ("Role: " + entry[0].strip().lower(), "Purpose: " + entry[1].strip().lower())
            if tup in rp_dict:
                rp_dict[tup]+=1
            else:
                rp_dict[tup]=1
    for entry in rp_gen:
        if len(entry)>=2:
            tup = ("Role: " + entry[0].strip().lower(), "Purpose: " + entry[1].strip().lower())
            if tup in rp_dict:
                rp_dict[tup]+=1
            else:
                rp_dict[tup]=1

    pa_con=match_purpose_attribute(purpose_nodes,data_attribute_nodes,text)
    pa_dict={}
    for entry in pa_con:
        if len(entry)>=2:
            tup = ("Purpose: " + entry[0].strip().lower(), "Attr: " + entry[1].strip().lower())
            if tup in pa_dict:
                pa_dict[tup]+=1
            else:
                pa_dict[tup]=1

    for entry in pa_gen:
        if len(entry)>=2:
            tup = ("Purpose: " + entry[0].strip().lower(), "Attr: " + entry[1].strip().lower())
            if tup in pa_dict:
                pa_dict[tup]+=1
            else:
                pa_dict[tup]=1

    print("\nRole-Purpose Connections:"+str(rp_dict))
   
    print("\nPurpose-Attribute Connections:"+str(pa_dict))

    def wrap_text(text, width=10):
        return '\n'.join(textwrap.wrap(text, width=width))








    G = nx.DiGraph()

    # Add role, purpose, and attribute nodes
    for node in role_nodes:
        G.add_node(node, type='role')
    for node in purpose_nodes:
        G.add_node(node, type='purpose')
    for node in data_attribute_nodes:
        G.add_node(node, type='attribute')

    print(str(G.nodes(data=True)))

    
    nodes_in_edges = set()

    # Add edges between roles and purposes
    for key in rp_dict.keys():
        role = key[0]
        purpose = key[1]
        if rp_dict[key] >= 1 or (role in role_nodes and purpose in purpose_nodes):
            if role not in role_nodes:
                role_nodes.append(role)
                G.add_node(role, type='role')
            if purpose not in purpose_nodes:
                purpose_nodes.append(purpose)
                G.add_node(purpose, type='purpose')
            G.add_edge(role, purpose)
            nodes_in_edges.add(role)
            nodes_in_edges.add(purpose)

    # Add edges between purposes and attributes
    for key in pa_dict.keys():
        purpose = key[0]
        attribute = key[1]
        if pa_dict[key] >= 1 or (purpose in purpose_nodes and attribute in data_attribute_nodes):
            if purpose not in purpose_nodes:
                purpose_nodes.append(purpose)
                G.add_node(purpose, type='purpose')
            if attribute not in data_attribute_nodes:
                data_attribute_nodes.append(attribute)
                G.add_node(attribute, type='attribute')
            G.add_edge(purpose, attribute)
            nodes_in_edges.add(purpose)
            nodes_in_edges.add(attribute)

    # Define 3D positions for nodes
    pos = {}
    for i, node in enumerate(role_nodes):
        pos[node] = (i * 2, 2 + random.uniform(0, 1), random.uniform(-1, 1))  # Add Z-coordinate

    for i, node in enumerate(purpose_nodes):
        pos[node] = (2 * i, 1, random.uniform(-1, 1))

    for i, node in enumerate(data_attribute_nodes):
        pos[node] = (1.5 * i, 0, random.uniform(-1, 1))

    # Extract node coordinates for Plotly
    x_nodes = [pos[node][0] for node in G.nodes()]
    y_nodes = [pos[node][1] for node in G.nodes()]
    z_nodes = [pos[node][2] for node in G.nodes()]

    # Extract edges and their coordinates
    edge_x = []
    edge_y = []
    edge_z = []
    for edge in G.edges():
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        edge_x += [x0, x1, None]  # None is to break the lines between edges
        edge_y += [y0, y1, None]
        edge_z += [z0, z1, None]

    # Create 3D scatter plot for nodes
    color_map = {
    'role': 'purple',      # You can change the color codes
    'purpose': 'blue',
    'attribute': 'green'
}

    # Create lists to store colors based on node types
    node_colors = []
    node_sentences=extract_node_sentences(text,G)
    # Loop through each node and assign color based on the node type
    node_colors = []
    for node in G.nodes():
        node_type = G.nodes[node].get('type', 'role')
        if node in nodes_in_edges:
            node_colors.append(color_map.get(node_type, 'gray'))
        else:
            node_colors.append('yellow')  # Color for nodes not in any edges

    # Create 3D scatter plot for nodes with different colors
    node_trace = go.Scatter3d(
        x=x_nodes, y=y_nodes, z=z_nodes,
        mode='markers+text',
        marker=dict(size=10, color=node_colors),  # Use node_colors for each node
        text=[wrap_text(node) for node in G.nodes()],  # Display wrapped node text on nodes
        hovertext=[node_sentences.get(node, '') for node in G.nodes()],  # Hover text from sentences
        textposition='middle center',  # Position the text in the center of the node
        textfont=dict(size=6, color='black'),  # Customize text size and color
    )
    

    # Create 3D line plot for edges
    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='black', width=1),
    )

    # Create the 3D figure
    fig = go.Figure(data=[edge_trace, node_trace])

    # Set layout for better visualization
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X-axis', showgrid=False, zeroline=False),
            yaxis=dict(title='Y-axis', showgrid=False, zeroline=False),
            zaxis=dict(title='Z-axis', showgrid=False, zeroline=False),
        ),
        title="3D Graph Visualization",
        showlegend=False
    )

    # Save the graph to a file
    timestamp = int(time.time())
    graph_path = f'graph_{timestamp}.html'
    fig.write_html(graph_path)

    # Show the figure in browser
    fig.show()
    # Print out the nodes and their types
    print("Nodes and their types:")
    for node, data in G.nodes(data=True):
        print(f"Node: {node}, Type: {data.get('type')}")

    # Print out the edges between nodes
    print("\nEdges (role -> purpose -> attribute):")
    for edge in G.edges():
        print(f"Edge from {edge[0]} to {edge[1]}")

    return graph_path


# Run the test
test_functions()
