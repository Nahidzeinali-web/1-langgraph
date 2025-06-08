# 🔍 LangChain Tools & LangGraph Integration Tutorial

This repository demonstrates how to integrate various tools from the `langchain_community` module—such as Wikipedia, YouTube, and Tavily search—as well as how to define and run custom tools and workflows using LangChain and LangGraph.

---

## 📦 Installation

```bash
pip install langchain langchain-community wikipedia youtube-search-python langgraph langchain-google-genai
```

---

## 🧠 Table of Contents

- [📘 Wikipedia Tool](#-wikipedia-tool)
- [📺 YouTube Search Tool](#-youtube-search-tool)
- [🌐 Tavily Search Tool](#-tavily-search-tool)
- [🧪 Custom Tools](#-custom-tools)
  - [🔢 Multiply Function](#multiply-function)
  - [🌤 Word Length Function](#word-length-function)
  - [📬 Placeholder: Gmail API Call](#placeholder-gmail-api-call)
- [🛠️ Common Mistakes & Fixes](#️-common-mistakes--fixes)
- [🔄 LangGraph Workflow](#-langgraph-workflow)
  - [✅ Sanity Check](#sanity-check)
  - [🧱 Build Simple Functions](#build-simple-functions)
  - [🔁 LangGraph Workflow with Functions](#langgraph-workflow-with-functions)
  - [🧠 LangGraph + Gemini LLM + Token Counter](#langgraph--gemini-llm--token-counter)
  - [📊 Visualize Workflow](#visualize-workflow)
- [🌐 LangGraph + LangChain + Gemini + HuggingFace Embeddings Tutorial](#-langgraph--langchain--gemini--huggingface-embeddings-tutorial)
- [👤 Author](#-author)
- [🧠 License](#-license)
---

## 📘 Wikipedia Tool

```python
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

api_wrapper = WikipediaAPIWrapper(top_k_results=5, doc_content_chars_max=500)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

print(wiki_tool.run({"query": "Elon Musk"}))
print(wiki_tool.run({"query": "RCB"}))
```

---

## 📺 YouTube Search Tool

```python
from langchain_community.tools import YouTubeSearchTool

tool = YouTubeSearchTool()
print(tool.run("sunny savita"))
print(tool.run("krish naik"))
```

---

## 🌐 Tavily Search Tool

```python
from langchain_community.tools.tavily_search import TavilySearchResults
import os

tool = TavilySearchResults(tavily_api_key=os.getenv("TAVILY_API_KEY"))

results = tool.invoke({"query": "RCB victory celebration"})
for r in results[:2]:
    print(f"Title: {r['title']}\nURL: {r['url']}\n")
```

---

## 🧪 Custom Tools

### 🔢 Multiply Function

```python
from langchain.agents import tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two integers."""
    return a * b

print(multiply.invoke({"a": 10, "b": 20}))
```

### 🔤 Word Length Function

```python
@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)

print(get_word_length.invoke({"word": "sunny"}))
```

### 📬 Placeholder: Gmail API Call

```python
@tool
def call_gmail_api(args):
    """Placeholder for Gmail API integration."""
    pass
```

---

## 🛠️ Common Mistakes & Fixes

| ❌ Issue                            | ✅ Solution                                                            |
|-----------------------------------|------------------------------------------------------------------------|
| `multiply.run(10, 20)`            | Use `multiply.invoke({"a": 10, "b": 20})`                              |
| `tool.run("query")`               | Use `.invoke({"query": "..."})` instead                                |
| Forgetting `@tool` decorator      | Add `@tool` from `langchain.agents` to define custom tools             |

---

## 🔄 LangGraph Workflow

### ✅ Sanity Check

```python
print("all ok")
```

### 🧱 Build Simple Functions

```python
def function1(input1):
    return input1 + " from first function"

def function2(input2):
    return input2 + " savita from second function"

def function3(input3):
    pass
```

### 🔁 LangGraph Workflow with Functions

```python
from langgraph.graph import Graph

workflow1 = Graph()
workflow1.add_node("fun1", function1)
workflow1.add_node("fun2", function2)
workflow1.add_edge("fun1", "fun2")
workflow1.set_entry_point("fun1")
workflow1.set_finish_point("fun2")

app = workflow1.compile()
```

### ▶️ Run and Stream

```python
print(app.invoke("hi this is sunny"))

for output in app.stream("hi this is rohit"):
    for key, value in output.items():
        print(f"Output from {key}:\n{value}\n")
```

### 🧠 LangGraph + Gemini LLM + Token Counter

```python
from langchain_google_genai import ChatGoogleGenerativeAI

def llm(input):
    model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')
    return model.invoke(input).content

def token_counter(input):
    tokens = input.split()
    return f"Total tokens: {len(tokens)}"
```

### 🔗 Create Workflow

```python
workflow2 = Graph()
workflow2.add_node("My_LLM", llm)
workflow2.add_node("Token_Counter", token_counter)
workflow2.add_edge("My_LLM", "Token_Counter")
workflow2.set_entry_point("My_LLM")
workflow2.set_finish_point("Token_Counter")

app = workflow2.compile()
print(app.invoke("Tell me about India's capital."))
```

### ⏳ Stream Output

```python
for output in app.stream("Details on Tata Enterprise."):
    for key, value in output.items():
        print(f"Output from {key}:\n{value}\n")
```

---

## 📊 Visualize Workflow

```python
from IPython.display import Image, display
display(Image(app.get_graph().draw_mermaid_png()))
```

---

---

## 🌐 LangGraph + LangChain + Gemini + HuggingFace Embeddings Tutorial

This tutorial walks you through how to build a simple intelligent agent using:

* Google Gemini (langchain_google_genai)
* HuggingFace Embeddings (langchain_huggingface)
* LangGraph state machines
* Vectorstore with Chroma
* Custom parsing using pydantic and langchain_core

---

### 📦 Step 1: Initialize a Chat Model

```python
from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')
output = model.invoke("hi")
print(output.content)
```

---

### 🧠 Step 2: Load Embedding Model
```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
len(embeddings.embed_query("hi"))
```

---

### 📁 Step 3: Load and Embed Text Documents

```python
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = DirectoryLoader("../data2", glob="./*.txt", loader_cls=TextLoader)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
new_docs = text_splitter.split_documents(documents=docs)

doc_string = [doc.page_content for doc in new_docs]
print(len(doc_string))

db = Chroma.from_documents(new_docs, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 3})
retriever.invoke("industrial growth of usa?")

```
---

### ⚙️ Step 4: Create a Pydantic Output Parser

```python
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser

class TopicSelectionParser(BaseModel):
    Topic: str = Field(description="selected topic")
    Reasoning: str = Field(description="Reasoning behind topic selection")

parser = PydanticOutputParser(pydantic_object=TopicSelectionParser)
print(parser.get_format_instructions())

```
---

### 🧪 Step 5: Understand Agent State

```python
Agentstate = {}
Agentstate["messages"] = []
Agentstate["messages"].append("hi how are you?")
Agentstate["messages"].append("what are you doing?")
Agentstate["messages"].append("i hope everything fine")

print(Agentstate["messages"][-1])

```
---

### 🧬 Step 6: Define Agent State Class

```python
import operator
from typing import Sequence, Annotated, TypedDict
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

```
---

### 🧠 Step 7: Build Classifier Node

```python
from langchain.prompts import PromptTemplate

def function_1(state: AgentState):
    question = state["messages"][-1]

    template = """
    Your task is to classify the given user query into one of the following categories: [USA, Not Related]. 
    Only respond with the category name and nothing else.

    User query: {question}
    {format_instructions}
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["question"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    chain = prompt | model | parser
    response = chain.invoke({"question": question})
    print("Parsed response:", response)

    return {"messages": [response.Topic]}

```
---

### 🔀 Step 8: Conditional Routing Logic

```python
def router(state: AgentState):
    last_message = state["messages"][-1]
    print("last_message:", last_message)

    if "usa" in last_message.lower():
        return "RAG Call"
    else:
        return "LLM Call"

```
---

### 🧠 Step 9: Define RAG Function

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def format_docs(docs):
    return "

".join(doc.page_content for doc in docs)


def function_2(state: AgentState):
    question = state["messages"][0]

    prompt = PromptTemplate(
        template="""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:""",
        input_variables=["context", "question"]
    )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    result = rag_chain.invoke(question)
    return {"messages": [result]}

```
---

### 💬 Step 10: Define Fallback LLM Function

```python
def function_3(state: AgentState):
    question = state["messages"][0]
    complete_query = f"Answer the following question using your real-world knowledge: {question}"
    response = model.invoke(complete_query)
    return {"messages": [response.content]}

```
---

### 🔁 Step 11: Build LangGraph Workflow

```python
from langgraph.graph import StateGraph, END

workflow = StateGraph(AgentState)

workflow.add_node("Supervisor", function_1)
workflow.add_node("RAG", function_2)
workflow.add_node("LLM", function_3)

workflow.set_entry_point("Supervisor")
workflow.add_conditional_edges("Supervisor", router, {
    "RAG Call": "RAG",
    "LLM Call": "LLM"
})

workflow.add_edge("RAG", END)
workflow.add_edge("LLM", END)

app = workflow.compile()

```
---

### 🚀 Step 12: Test the Application

python
state = {"messages": ["what is a GDP of usa?"]}
result = app.invoke(state)


Try other messages too:

python
state = {"messages": ["can you tell me the industrial growth of world's poor economy?"]}
result = app.invoke(state)

```
---

## 👤 Author

Made with 💡 by [@NahidZeinali-web](https://github.com/Nahidzeinali-web)

---

## 🧠 License

This project is licensed under the [MIT License](LICENSE).
