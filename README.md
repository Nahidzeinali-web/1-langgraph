# 🔍 LangChain Tools Integration Tutorial

This repository demonstrates how to use various tools from the `langchain_community` module, including Wikipedia, YouTube, Tavily search tools, and how to define your own custom tools with LangChain.

---

## 📦 Installation

```bash
pip install langchain langchain-community wikipedia youtube-search-python
```

---

## 🧠 Table of Contents

- [📘 Wikipedia Tool](#-wikipedia-tool)
- [📺 YouTube Search Tool](#-youtube-search-tool)
- [🌐 Tavily Search Tool](#-tavily-search-tool)
- [🧪 Custom Tool: Multiply Function](#-custom-tool-multiply-function)
- [🧮 Custom Tool: Word Length](#-custom-tool-word-length)
- [📬 Placeholder Tool: Gmail API Call](#-placeholder-tool-gmail-api-call)
- [🛠️ Common Mistakes & Fixes](#️-common-mistakes--fixes)

---

## 📘 Wikipedia Tool

**Purpose**: Answer general knowledge questions using Wikipedia.

```python
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

api_wrapper = WikipediaAPIWrapper(top_k_results=5, doc_content_chars_max=500)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

print(wiki_tool.name)          # 'wikipedia'
print(wiki_tool.description)
print(wiki_tool.args)

# Run the tool
print(wiki_tool.run({"query": "Elon Musk"}))
print(wiki_tool.run({"query": "RCB"}))
```

> ✅ **Tip**: Ignore the BeautifulSoup parser warning for basic use.

---

## 📺 YouTube Search Tool

**Purpose**: Find videos by person name on YouTube.

```python
from langchain_community.tools import YouTubeSearchTool

tool = YouTubeSearchTool()

print(tool.name)               # 'youtube_search'
print(tool.description)

# Run the tool
print(tool.run("sunny savita"))
print(tool.run("krish naik"))
```

> 🔢 To limit results, use: `"krish naik, 3"`

---

## 🌐 Tavily Search Tool

**Purpose**: Perform real-time web search.

```python
from langchain_community.tools.tavily_search import TavilySearchResults
import os

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
tool = TavilySearchResults(tavily_api_key=TAVILY_API_KEY)

results = tool.invoke({"query": "what happened in RCB victory celebration?"})

for r in results[:2]:
    print(f"Title: {r['title']}\nURL: {r['url']}\n")
```

---

## 🧪 Custom Tool: Multiply Function

```python
from langchain.agents import tool

@tool
def multiply(a: int, b: int) -> int:
    """this tool is for the multiplication"""
    return a * b

print(multiply.invoke({"a": 10, "b": 20}))  # Output: 200

# Metadata
print(multiply.name)
print(multiply.description)
print(multiply.args)
```

---

## 🧮 Custom Tool: Word Length

```python
@tool
def get_word_length(word: str) -> int:
    """this function is calculating the length of the word"""
    return len(word)

print(get_word_length.invoke({"word": "sunny"}))  # Output: 5

# Metadata
print(get_word_length.name)
print(get_word_length.description)
print(get_word_length.args)
```

---

## 📬 Placeholder Tool: Gmail API Call

```python
@tool
def call_gamil_api(args):
    """this is my gmail API calling function"""
    pass
```

---

## 🛠️ Common Mistakes & Fixes

| ❌ Problem                          | ✅ Fix                                                               |
|-----------------------------------|----------------------------------------------------------------------|
| `multiply.run(10, 20)`            | Use `multiply.invoke({"a": 10, "b": 20})` after applying `@tool`     |
| `tool.run("query")`               | For most LangChain tools, use `.invoke({"query": "..."})` instead    |
| Using regular functions as tools  | You **must** decorate them with `@tool` from `langchain.agents`     |

---

## 📎 Author

Made by [@NahidZeinali-web](https://github.com/Nahidzeinali-web) using LangChain & Python 🐍

---

## 🧠 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
