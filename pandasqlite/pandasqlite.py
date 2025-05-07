import json
from typing import List
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
import pandas as pd
import pickle
import re
from hashlib import md5
import os
import logging

from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.schema import CreateTable

logger = logging.getLogger("pandasqlite")

client = None
llm_is_sane = False

CACHE_DIR = os.getenv("PANDASQLITE_CACHE_DIR", "cache")


def watsonxai(input: str):
    logger.debug("START: watsonxai")
    global client
    wxai_project_id = os.getenv("WXAI_PROJECT_ID")
    wxai_api_key = os.getenv("WXAI_API_KEY")
    if client is None:
        credentials = Credentials(
            url="https://us-south.ml.cloud.ibm.com/",
            api_key=wxai_api_key
        )
        client = APIClient(credentials)
    model = ModelInference(
        # model_id="ibm/granite-3-2-8b-instruct",
        model_id="meta-llama/llama-3-405b-instruct",
        api_client=client,
        project_id=wxai_project_id,
        params={
            "max_new_tokens": 4096,
            "time_limit": 100000,
            "decoding_method": "greedy",
            "min_new_tokens": 10,
            "stop_sequences": ["\n\n"],
            "repetition_penalty": 1
        }
    )
    logger.debug(input)
    result = model.generate_text("input:" + input)
    result = result.split("\n\n")[0]
    logger.debug("output:" + result)
    logger.debug("END: watsonxai")
    return result


def ingest(dfs: List[pd.DataFrame] | str, llm_callback=watsonxai):
    """
    Newly ingests a set of data frames or looks up previous ingestion results by hash using built-in cache. If the exact
    same data is provided twice, a lookup from cache triggers also by computing the hash over the data.
    :param dfs: List of Pandas Dataframes or a single String representing the hash value result of a previous ingestion.
    :param llm_callback: Optional custom callback to resolve language model requests through.
    Defaults to pandasqlite.watsonxai.
    :return: triple of (1) ingestion result, (2) db engine, (3) hash
    """
    logger.debug("START: ingest")
    # step 0: sanity check
    sanity_check(llm_callback)

    # use or compute hash over dfs
    if isinstance(dfs, str):
        dfs_hash = dfs
    else:
        logger.debug("Computing hash...")
        hash_object = md5("".join([df.to_string() for df in dfs]).encode())
        dfs_hash = hash_object.hexdigest()
    engine = create_engine('sqlite:///cache/' + dfs_hash + '.db', echo=False)

    # check if ingestion is pickled
    logger.debug("Checking cache availability...")
    os.makedirs("cache", exist_ok=True)
    if os.path.exists(f'{CACHE_DIR}/{dfs_hash}.pkl'):
        logger.info("Loading ingestion from cache...")
        with open(f'{CACHE_DIR}/{dfs_hash}.pkl', 'rb') as f:
            ingestion_results = pickle.load(f)
            return ingestion_results, engine, dfs_hash

    if isinstance(dfs, str):
        raise "Hash not found."

    ingestion_results = []
    for df in dfs:
        # ensure compatibility of dataframe column names
        df.columns = [re.sub('[^0-9a-zA-Z]+', '_', col) for col in df.columns]

        # compute hash over df
        logger.debug("Computing hash...")
        hash_object = md5(df.to_string().encode())
        df_hash = hash_object.hexdigest()

        logger.info("Ingesting from scratch...")
        ingestion_result = {
            "context": None,
            "column_types": None,
            "value_format": None,
            "column_descriptions": None,
            "enum_descriptions": None,
            "sql_curriculum": None,
            "hash": df_hash,
            "ddl": None
        }

        logger.info("Creating database for dataframe and extracting DDL...")
        sqlite_connection = engine.connect()
        df.to_sql(ingestion_result["hash"], sqlite_connection, if_exists='replace')
        metadata = MetaData()
        metadata.reflect(bind=engine)
        table_to_generate_ddl_for = Table(ingestion_result["hash"], metadata, autoload_with=engine)
        ddl = CreateTable(table_to_generate_ddl_for).compile(engine)
        ingestion_result["ddl"] = ddl.string
        sqlite_connection.close()

        reverse_column_lookup = {column.lower(): column for column in df.columns}

        logger.info("Probing data samples...")
        ingestion_result["value_format"] = snapshot_data(df)

        logger.info("Estimating column types...")
        prompt = "You are a data scientists who has to estimate the data type of input tables. You respond with JSON format. You will now see a dictionary where the keys are the column names, and a few example values in an array as the value. Return a dictionary where the keys are again the column names, but put the datatype in the value. You can only select from the following data types in your response: TEXT, NUMBER or ENUM.\n\n" \
                 "INPUT:{\"sex\":[\"m\",\"m\",\"f\"]}\n" \
                 "OUTPUT:{\"sex\":\"ENUM\"}\n\n" \
                 "INPUT:"
        result = llm_callback(prompt + json.dumps(snapshot_data(df), separators=(',', ':')) + "\nOUTPUT:")
        ingestion_result["column_types"] = json.loads(result)

        logger.info("Generating column descriptions...")
        prompt = "You are a data scientists who has to generate descriptions for columns of input tables. You respond with JSON format. Data must not be revealed. You will now see a dictionary where the keys are the column names, and a few example values in an array as the value. Return a dictionary where the keys are again the column names, but put the generated descriptions in the value.\n\n" \
                 "INPUT:{\"cst_num\":[\"14\",\"12\",\"28\"],\"sex\":[\"male\",\"female\",\"male\"]}\n" \
                 "OUTPUT:{\"cst_num\":\"The number of customers.\",\"sex\":\"The gender of the customers.\"}\n\n" \
                 "INPUT:"
        result = llm_callback(prompt + json.dumps(snapshot_data(df), separators=(',', ':')) + "\nOUTPUT:")
        ingestion_result["column_descriptions"] = json.loads(result)

        logger.info("Generating enum descriptions...")
        prompt = "You are a data scientists who has to generate descriptions for column class/enum values of input tables. You respond with JSON format. You will now see a dictionary where the keys are the column names, and the class/enum values in an array as the value. Return a dictionary where the keys are again the column names, but add another dictionary for the generated class/enum descriptions as the value.\n\n" \
                 "INPUT:{\"gender\":[\"male\",\"female\"]}\n" \
                 "OUTPUT:{\"gender\":{\"male\":\"The person is of male gender.\",\"female\":\"The person is of female gender.\"}}\n\n" \
                 "INPUT:"
        enum_values = distinct_enum_values(
            df, [reverse_column_lookup[k] for k, v in ingestion_result["column_types"].items() if v == "ENUM"]
        )
        enum_values = {key.lower(): [str(v) for v in enum_values[key].tolist()] for key in enum_values.keys()}
        result = llm_callback(
            prompt + json.dumps(enum_values, separators=(',', ':')).replace("NaN", "\"NaN\"") + "\nOUTPUT:"
        )
        ingestion_result["enum_descriptions"] = json.loads(result)

        logger.info("Generating sql curriculum...")
        prompt = "You are a data scientists who has to generate natural language questions and their corresponding SQLLite solutions based on a description in JSON for an input table. You respond with JSON format, an array with comma-separated objects. You will now see a dictionary where the keys are the column names, and a few example values in an array as the value. Return up to 7 examples for natural language questions and their corresponding SQLLite solution. Make sure to increase the complexity of the questions beginning from very simple (1) to very hard (5). Infer the types from the data as provided, so do not turn string values into numbers or booleans, or vice versa. For aggregation functions wrap the columns in parentheses, e.g. MIN(column_name), etc.\n\n" \
                 "INPUT:{\"sex\":[\"male\",\"female\"],\"survived\":[\"0\",\"1\"]}\n" \
                 "OUTPUT:[{\"question\":\"Give me the different genders of the passengers.\",\"sql\":\"SELECT DISTINCT sex FROM table WHERE survived=\\\"1\\\" ORDER BY sex ASC\"}]\n\n" \
                 "INPUT:"
        result = llm_callback(prompt + json.dumps(snapshot_data(df), separators=(',', ':')) + "\nOUTPUT:")
        try:
            if not result.strip().startswith("["):
                result = "[" + result
            ingestion_result["sql_curriculum"] = json.loads(result)
            for question in ingestion_result["sql_curriculum"]:
                question["sql"] = question["sql"].replace("table", "'" + ingestion_result["hash"] + "'").strip()
        except json.decoder.JSONDecodeError as e:
            logger.error(e)

        ingestion_results.append(ingestion_result)

    logger.info("Cashing ingestion result...")
    with open(f'{CACHE_DIR}/{dfs_hash}.pkl', 'wb') as f:
        pickle.dump(ingestion_results, f)

    logger.debug("END: ingest")
    return ingestion_results, engine, dfs_hash


def text2sql(input: str, ingestions, llm_callback=watsonxai):
    logger.debug("START: text2sql")
    prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>" + "\n"

    # SCHEMA
    prompt += "[SCHEMA]" + "\n"
    for ingestion in ingestions:
        prompt += ingestion["ddl"] + "\n"
    prompt += "[/SCHEMA]" + "\n\n"

    # VALUE FORMAT
    prompt += "[VALUE_FORMAT]" + "\n"
    for ingestion in ingestions:
        prompt += json.dumps(ingestion["value_format"], separators=(',', ':')) + "\n"
    prompt += "[/VALUE_FORMAT]" + "\n\n"

    # ENUM DESCRIPTIONS
    prompt += "[ENUMS]" + "\n"
    prompt += "Descriptions of values of enum columns in the schema." + "\n\n"
    for ingestion in ingestions:
        for key, value in ingestion["enum_descriptions"].items():
            for key2, value2 in value.items():
                prompt += key + "." + key2 + ": " + value2 + "\n"
            prompt += "\n"
    prompt += "[/ENUMS]" + "\n\n"

    # CONTEXT
    prompt += "[DOCUMENTATION]" + "\n"
    for ingestion in ingestions:
        if "context" in ingestion and ingestion["context"] is not None:
            prompt += ingestion["context"] + "\n"
    else:
        prompt += "No documentation provided.\n"
    prompt += "[/DOCUMENTATION]" + "\n\n"

    # INSTRUCTION
    prompt += "[INSTRUCTION]" + "\n"
    prompt += "Given the above schema of the database at [SCHEMA], the example of the values format of the " \
              "columns at [VALUE_FORMAT], the definitions of the enums at [ENUMS], the documentation at " \
              "[DOCUMENTATION], and the below question [QUESTION] translate the question into a valid SQL " \
              "statement compliant to SQLite. Format the output using the Markdown language for the SQL code. " \
              "Generate only the SQL code without any further text, i.e. COMMENTS are STRICTLY FORBIDDEN. " \
              "When computing a correlation, avoid using CORR or AVG." + "\n"
    prompt += "[/INSTRUCTION]" + "\n\n"

    # QUESTIONS
    for ingestion in ingestions:
        if "sql_curriculum" in ingestion and ingestion["sql_curriculum"] is not None:
            for i in range(0, len(ingestion["sql_curriculum"])):
                prompt += "[QUESTION]" + "\n"
                prompt += ingestion["sql_curriculum"][i]["question"] + "\n"
                prompt += "[/QUESTION]" + "\n"
                prompt += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>" + "\n"
                prompt += "```sql" + "\n"
                prompt += ingestion["sql_curriculum"][i]["sql"] + "\n"
                prompt += "```" + "\n"
                prompt += "<|eot_id|><|start_header_id|>user<|end_header_id|>" + "\n"

    prompt += "[QUESTION]" + "\n"
    prompt += input + "\n"
    prompt += "[/QUESTION]" + "\n"
    prompt += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>" + "\n"
    sql = llm_callback(prompt)
    result = sql.split("```sql")[1].split("\n```")[0].strip()
    result = result.replace("table", "'" + ingestion["hash"] + "'").strip()
    logger.debug("generated sql: " + result)
    logger.debug("END: text2sql")
    return result


def sanity_check(llm_callback=watsonxai):
    logger.debug("START: sanity_check")
    global llm_is_sane
    if llm_is_sane:
        return
    # test model for sanity
    logger.info("Testing model sanity...")
    try:
        response = watsonxai(
            "<system>If you are a language model, write 'YES' as the next token, followed by 2 empty lines.<system>"
        )
    except:
        logger.info("××× FAILED ×××")
        logger.info(
            "Token generation failed. You need to provide a reasonable " +
            "language model to use this software. See documentation."
        )
        exit(0)
    if response != "YES" and response != "'YES'":
        logger.info("××× FAILED ×××")
        logger.info(
            "Your provided model responded with '" + response + "', when it was 'YES' that was asked for. You " +
            "need to provide a reasonable language model to use this software. See documentation."
        )
        exit(0)
    logger.info("✓✓✓ PASSED ✓✓✓\n")
    llm_is_sane = True
    logger.debug("END: sanity_check")


def distinct_enum_values(df, enum_columns):
    logger.debug("START: distinct_enum_values")
    result = df[enum_columns].apply(pd.Series.unique)
    logger.debug("END: distinct_enum_values")
    return result


def snapshot_data(df):
    logger.debug("START: snapshotdata")
    # Create an empty dictionary
    result = {}

    # Iterate over each column in the DataFrame
    for column in df.columns:
        # Select 5 random samples from the column
        samples = df[column].sample(5)
        # Add the column name and the samples to the dictionary
        result[column.lower()] = samples.tolist()
    logger.debug("END: snapshotdata")
    return result
