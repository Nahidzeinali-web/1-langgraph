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
  - [🔤 Word Length Function](#word-length-function)
  - [📬 Placeholder: Gmail API Call](#placeholder-gmail-api-call)
- [🛠️ Common Mistakes & Fixes](#️-common-mistakes--fixes)
- [🔄 LangGraph Workflow](#-langgraph-workflow)
  - [✅ Sanity Check](#sanity-check)
  - [🧱 Build Simple Functions](#build-simple-functions)
  - [🔁 LangGraph Workflow with Functions](#langgraph-workflow-with-functions)
  - [🧠 LangGraph + Gemini LLM + Token Counter](#langgraph--gemini-llm--token-counter)
  - [📊 Visualize Workflow](#visualize-workflow)
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

## 👤 Author

Made with 💡 by [@NahidZeinali-web](https://github.com/Nahidzeinali-web)

---

## 🧠 License

This project is licensed under the [MIT License](LICENSE).
