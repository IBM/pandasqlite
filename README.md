# PandaSQLite - Query CSV files in Natural Language with precision of SQL
Discover the power of Light-weight Text-2-SQL on Pandas Data Frames, a user-friendly Python module that transforms natural language questions into SQL queries for effortless data analysis. With the pandasqlite library, you can load CSV files into Pandas DataFrames, ingest them, and query data using simple questions like "What are the categories for products sold in Italy?" or "Who are the top 10 customers by turnover last month?" The tool generates and executes SQL queries instantly, returning results in JSON format for seamless integration into your workflows. Perfect for data analysts and developers, this lightweight solution eliminates the need for complex SQL knowledge or database setup, making it ideal for quick insights from sales, customer, or inventory data. Optimized for simplicity, it streamlines data exploration, boosts productivity, and empowers users to uncover valuable insights with minimal code, enhancing efficiency in any data-driven project.

## Installation 

1. Clone repository
```bash
git clone git@github.com:IBM/pandasqlite.git
```

2. Install PandaSQLite
```bash
cd pandasqlite
pip install .
```

3. Set environment variables:

- `PANDASQLITE_CACHE_DIR` - (optional) Set the cache directory location

4. Choose language model backend (see section below for more details):
   1. Ollama (default, local)
        ```
        # Manual installation required! 
        # Download from https://ollama.com/
        ```
   2. watsonx.ai (hosted)
      - Get [watsonx.ai](https://www.ibm.com/products/watsonx-ai) API key
      - Set environment variables:
        - `WXAI_PROJECT_ID` - Set to your watsonx.ai project ID
        - `WXAI_API_KEY` - Set to your watsonx.ai API key 
   3. Custom language model

## Using PandaSQLite in Python:

```python
import json
import pandas as pd
from pandasqlite import pandasqlite as pdsql

# load CSV as pandas dataframe(s)
df1 = pd.read_csv("my.csv")
df2 = ...

# ingest dataframe(s)
ingestion, db, _ = pdsql.ingest([df1, df2, ...])

# ask some questions
for question in [
    "Show the categories for products sold in Italy.",
    "Return the top 10 customers with highest turnover last month, sorted alphabetically by last name.",
    "What's the average number of items sold per purchase?",
    "Generate an interesting query."
]:
    sql = pdsql.text2sql(question, ingestion)  # generate query
    result = pd.read_sql(sql, db)              # execute query
    print(question)
    print(json.dumps(result.to_json()) + "\n")
```

You can also take a look at this [example](https://github.com/IBM/PandaSQLite/blob/main/test.py).

## Choosing a Language Model Backend

### Ollama Language Model (Local)
By default, PandaSQLite attempts to connect to a local Ollama Server. You will have to download and install this software from https://ollama.com/.


### watsonx.ai Language Model (Remote/Hosted)
If you do not have the resources to run a language model with Ollama locally, you may find this hosted option attractive:
- Get [watsonx.ai](https://www.ibm.com/products/watsonx-ai) API key
  - `WXAI_PROJECT_ID` - Set to your watsonx.ai project ID
  - `WXAI_API_KEY` - Set to your watsonx.ai API key

```python
pdsql.ingest([df1, df2, ...], pdsql.watsonxai)              # ingest with watsonx.ai

sql = pdsql.text2sql(question, ingestion, pdsql.watsonxai)  # generate query with watsonx.ai
``` 

### Custom Language Model
You can even plug-in a custom language model callback function as a parameter:

```python
def my_model_callback(input):
    # resolve input string by call to local model or external service
    output = ...
    return output

pdsql.ingest([df1, df2, ...], my_model_callback)              # ingest with custom model

sql = pdsql.text2sql(question, ingestion, my_model_callback)  # generate query with custom model
``` 

# How to cite
> Daniel Karl I. Weidele and Gaetano Rossiello. PandaSQLite: Light-weight Text-2-SQL on Pandas Data Frames in Python. GitHub, https://github.com/IBM/pandasqlite. 2025.


