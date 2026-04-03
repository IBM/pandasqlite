import logging, sys, os, json
import pandas as pd
from pandasqlite import pandasqlite as pdsql

logging.basicConfig(
    # level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("pandasqlite_test.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

if __name__ == "__main__":

    logger = logging.getLogger("pandasqlite_test")
    logger.setLevel(logging.INFO)

    # llm_callback = pdsql.ollama  # requires Ollama installed locally
    llm_callback = pdsql.watsonxai  # requires watsonx.ai API keys in environment
    # llm_callback = [to use custom llm, provide a function that takes a string instruction and returns the generated output string]

    # datasets with descriptions
    datasets = [
        {
            "file": "test_data",
            "questions": [
                # "what's the most popular product?",
                # "perform an interesting analysis"
                # "for each program for which the new runtime (P Elapsed by Claim) increased by more than 10 % compared to the old runtime (B Elapsed by Claim) (just use these 2 columns here as a filter), recursively list all subprograms that it called sorted by their runtime descendingly. keep in mind that the subprograms can call each other recursively."
                "list the recursive table (up to 5 levels) for calling programs to called programs starting from programs for which the new runtime (P Elapsed by Claim) increased by more than 10 % compared to the old runtime (B Elapsed by Claim) (just use these 2 columns here as a filter)."
            ]
        }
    ]

    for dataset in datasets:
        logger.info("Processing " + dataset["file"] + "...")
        dfs = []
        if os.path.isfile(dataset["file"]):
            # load single CSV file
            dfs.append(pd.read_csv(dataset["file"]))
        else:
            # load multiple CSV files from folder
            for f in os.listdir(dataset["file"]):
                dfs.append(pd.read_csv(dataset["file"] + "/" + f))

        # ingest and ask some questions
        ingestion, db, hash = pdsql.ingest(dfs, llm_callback)
        # OR lookup via previous hash: ingestion, db, hash = pdsql.ingest("58dade241cb167f920eb090e1deed5dc")
        for question in dataset["questions"]:
            logger.info("Question:")
            logger.info("\t\t" + question)
            sql = pdsql.text2sql(question, ingestion, llm_callback)
            # sql = "WITH RECURSIVE program_calls AS ( SELECT      Calling_Program_Name,      Called_Program_Name,      0 AS level   FROM      \"63de38fa7011a9403e86707babf5d0ee\" WHERE      Calling_Program_Name IN ( SELECT       Program_Name FROM          \"5ea8200da0f0d5a3b26cf59f8245d9e0\" WHERE          (P_Elapsed_time_by_Claim_at_RCNCP032_ms_ > 1.1 * B_Elapsed_time_by_Claim_at_RCNCP032_ms_) ) UNION ALL SELECT      pc.Calling_Program_Name,      e.Called_Program_Name,      level + 1 FROM program_calls pc   JOIN      \"63de38fa7011a9403e86707babf5d0ee\" e ON pc.Called_Program_Name = e.Calling_Program_Name WHERE pc.level < 5 ) SELECT    Calling_Program_Name,    Called_Program_Name,    level FROM    program_calls;"
            logger.info("Generated SQL:")
            logger.info("\t\t" + sql.replace("\n", " "))
            result = pd.read_sql(sql, db)
            logger.info("Result (5 sample rows):")
            if result.shape[0] > 5:
                logger.info("\t\t" + json.dumps(result.sample(5).to_json()) + "\n")
            else:
                logger.info("\t\t" + json.dumps(result.to_json()) + "\n")

        # export dataframe by index position (e.g. 2nd and 4th)
        # dfs_exp = pdsql.export([ingestion[1], ingestion[3]], db)
        # logger.info(f'Exported {len(dfs_exp)} dataframes.')

        # export dataframe by hash
        # dfs_exp = pdsql.export([i for i in ingestion if i["hash"] == "1eeb622f464ab027744d7c5dd96a6826"], db)
        # logger.info(f'Exported {len(dfs_exp)} dataframes.')
