# TOOLS AND THE BOT 

# Create the Document Pipeline 
# Add to the schema 
# Edit the Prompt 




# Imports 
from openai import OpenAI
import csv
from shipment import shipments
import json
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import json 
from pathlib import Path
from dotenv import load_dotenv
import os 

# Load All the Enviornemtn Variables 
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
kb = os.getenv("KB")



# Create Client
client = OpenAI(api_key=openai_api_key)
embeddings = OpenAIEmbeddings(api_key=openai_api_key)



# Document Pipeline 

def load_document(file_path):
    """Load Document based on the file path"""
    file_extension = os.path.splitext(file_path)[1]

    if file_extension.lower() == ".pdf":
        loader = PyPDFLoader(file_path)
    elif file_extension.lower() == ".txt":
        loader = TextLoader(file_path)
    elif file_extension.lower() in [".docx", "doc"]:
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported File Format")

    return loader.load()


def create_vector_db_from_document(file_path):
    # Load the File
    document = load_document(file_path)
    # Split it 
    text_spliter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_spliter.split_documents(document)
    # Create the Vector Embeddings 
    db = FAISS.from_documents(docs, embeddings)
    return db


def generate_answer_from_document(query: str, k: int = 4):
    database = create_vector_db_from_document(kb)
    docs = database.similarity_search(query=query, k=k)
    docs_page = "".join([d.page_content for d in docs])
    return docs_page




def file_return(product_id):
    with open("return.csv","a",newline="") as file:
        writer = csv.writer(file)
        writer.writerow([product_id,"Pending"])
        return f"Return for product ID {product_id} filed successfully"
    


def get_shipment_status(product_id):
    if product_id in shipments:
        data = shipments[product_id]
        return f"Status: {data['status']}, Expected Delivery:{data['expected_delivery']},Product: {data['carrier']}"
    else:
        return f"Invalid Product ID, No Product Found In the Database"
    



agent_tools= [
    {
        "type": "function",
        "function": {
            "name": "file_return",
            "description": "File a return request for a product using the product ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "The unique Product ID to file the return for."
                    }
                },
                "required": ["product_id"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_shipment_status",
            "description": "Get the current shipment status for a given product.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "The unique Product ID to check shipment status."
                    }
                },
                "required": ["product_id"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_answer_from_document",
            "description": "Retrieve information about Spacebuddy And it's Produtcsfrom the company knowledge base.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The question about Spacebuddy that needs to be answered from the knowledge base",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    }
]



system_message = """
<system_prompt>
YOU ARE **TRENT**, A FRIENDLY, EFFICIENT AI SUPPORT AGENT WORKING FOR **SPACE BUDDY**, AN E-COMMERCE STORE THAT SELLS GALAXY PROJECTORS AND SPACE-THEMED PRODUCTS.

YOUR MISSION IS TO ASSIST CUSTOMERS WITH THE FOLLOWING TASKS USING THE PROVIDED TOOLS:

### 🛠️ TOOLS:

1. **file_return**
   - **Purpose**: File a return request for a purchased product.
   - **Parameters Required**: `product_id` (Provided by the user)
   - **Use Case**: ONLY call this tool after the user has provided the Product ID and has expressed the intent to return a product.

2. **get_shipment_status**
   - **Purpose**: Get the current shipment status of a customer’s order.
   - **Parameters Required**: `product_id` (Provided by the user)
   - **Use Case**: ONLY call this tool when the user asks about their shipment or delivery status AND has provided the Product ID.

3. **generate_answer_from_document**
   - **Purpose**: Retrieve accurate information about Space Buddy and its products from the company's official documentation.
   - **Parameters Required**: `query` (The user's question about Space Buddy)
   - **Use Case**: Call this tool when the user asks:
     - About company policies, product details, specifications, materials, or any general inquiry not related to shipment or returns.
   - **How to Use**: Pass the user's full question as the `query`. After receiving the tool response, use it to answer the user naturally, without mentioning the tool or the process.

### ⚙️ RULES:

- If a **product_id is not provided** for returns or shipment status inquiries, DO NOT call any tool. INSTEAD, politely ask:
  **"Could you please provide your Product ID so I can assist you?"**

- ALWAYS call the appropriate tool **before responding** to the user.  
  NEVER promise action or give information until the tool’s output is received.

- When using **generate_answer_from_document**, DO NOT say you're "checking" or "retrieving" information. Just answer directly once the tool provides the information.

- Speak like a **normal, professional, and friendly human assistant**. Avoid "space-themed" language.

- Be concise, accurate, and helpful in every response.

- **NEVER invent information**. Only reply based on tool outputs.

### 💬 EXAMPLES:

**User**: I want to return my galaxy lamp.  
**Trent**: (*[First call file_return tool after getting Product ID]*)  
**Response After Tool Call**: Your return request has been successfully filed. You'll receive a confirmation email shortly!

---

**User**: When will my space projector arrive?  
**Trent**: (*[First call get_shipment_status tool after getting Product ID]*)  
**Response After Tool Call**: Your order is currently in transit and should arrive within the next 3-5 business days.

---

**User**: What materials are used in your galaxy projectors?  
**Trent**: (*[First call generate_answer_from_document tool with query]*)  
**Response After Tool Call**: Our galaxy projectors are made from durable ABS plastic and high-grade acrylic to ensure longevity and quality.

---

**User**: What's Space Buddy’s return policy?  
**Trent**: (*[First call generate_answer_from_document tool with query]*)  
**Response After Tool Call**: You can return products within 30 days of delivery as long as they are unused and in their original condition.

### 🤝 PERSONALITY:
Trent is warm, polite, professional, and approachable.  
You sound like a knowledgeable and efficient customer support agent — always focused on providing accurate and helpful responses without unnecessary delays.  
Be clear, helpful, and friendly — just like a real human assistant.  
NEVER mention internal tools or processes to users.  
NEVER guess or create information; always rely strictly on tool outputs.

</system_prompt>



"""

conversation_memory = [{"role":"system","content":system_message}]


# Chat
def chat(user_input):
    conversation_memory.append({"role":"user","content":user_input})
    completions_1 = client.chat.completions.create(
        model="gpt-4o",
        messages=conversation_memory,
        tools=agent_tools,
        temperature=0.6
        
    )

    response_1 = completions_1.choices[0].message
    


    if response_1.tool_calls:


        conversation_memory.append({
        "role": "assistant",
        "content": response_1.content,
        "tool_calls": response_1.tool_calls
    })
        
        def call_function(name, args):
            if name == "file_return":
                return file_return(**args)
            elif name == "get_shipment_status":
                return get_shipment_status(**args)
            elif name == "generate_answer_from_document":
                return generate_answer_from_document(**args)


        for tool_call in response_1.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)

                result = call_function(name, args)
                conversation_memory.append(
                {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)}
            )
                
        completions_2 = client.chat.completions.create(
            model="gpt-4o",
            messages=conversation_memory,
            temperature=0.5
        )

        response_2 = completions_2.choices[0].message
        conversation_memory.append(
        {"role": "assistant", "content": response_2.content})
        return response_2.content 
    
    else:
        conversation_memory.append({
            "role": "assistant",
            "content": response_1.content
        })
        return response_1.content


#################################################################################################################################################
############################################################## TELEGRAM #######################################################################
import telebot 

bot = telebot.TeleBot(token="7323295143:AAEpb4LW46WUk8eZhmxn_MLK2P3iOowyzxg")


# The Main Command 
@bot.message_handler(func= lambda msg: True)
def telegram_bot(message):
    user_query = message.text 
    try:
        response = chat(user_input=user_query)
        bot.send_message(message.chat.id,response,parse_mode="Markdown")
    except Exception as e:
        bot.send_message(bot.chat_id,"An Unexpected Error Happened Please Try Later", parse_mode="Markdown")



bot.infinity_polling()
        
