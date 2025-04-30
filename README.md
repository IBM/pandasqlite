# PandaSQLite
Light-weight Text-2-SQL on Pandas Data Frames

## Installation 

1. Clone repository
```bash
git clone git@github.com:IBM/PandaSQLite.git
```

2. Install PandaSQLite
```bash
cd pandasqlite
pip install .
```

3. Choose language model:

- Get your watsonx.ai API key from https://www.ibm.com/products/watsonx-ai, then continue with step 4.
- OR use a custom language model (see below)

4. Set environment variables:
- `WXAI_PROJECT_ID` - Set to your watsonx.ai project ID
- `WXAI_API_KEY` - Set to your watsonx.ai API key

## Using PandaSQLite in Python:

```python
import json
import pandas as pd
import pandasqlite as pdsql

# load CSV as pandas dataframe(s)
df1 = pd.read_csv("my.csv")
df2 = ...

# ingest dataframe(s)
ingestion, db = pdsql.ingest([df1, df2, ...])

# ask some questions
for question in [
    "What is the survival rate?",
    "Generate an interesting query"
]:
    sql = pdsql.text2sql(question, ingestion)  # generate query
    result = pd.read_sql(sql, db)              # execute query
    print(question)
    print(json.dumps(result.to_json()) + "\n")
```

You can also take a look at this [example](https://github.com/IBM/PandaSQLite/blob/main/test.py).

## Custom Language Model
Not ready to use watsonx.ai? You can plug-in a custom language model callback function as a parameter:

```python
def my_model_callback(input):
    # resolve input string by call to local model or external service
    output = ...
    return output

pdsql.ingest([df1, df2, ...], my_model_callback)              # ingest with custom model

sql = pdsql.text2sql(question, ingestion, my_model_callback)  # generate query with custom model
``` 

# How to cite
> Daniel Karl I. Weidele and Gaetano Rossiello. PandaSQLite: Light-weight Text-2-SQL on Pandas Data Frames in Python. GitHub, https://github.com/IBM/PandaSQLite. 2025.


