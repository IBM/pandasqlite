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

    # datasets with descriptions
    datasets = [
        {
            "file": "sample_data",
            "questions": [
                "what's the most popular product?",
                "perform an interesting analysis"
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
        ingestion, db, hash = pdsql.ingest(dfs)
        # OR lookup via previous hash: ingestion, db, hash = pdsql.ingest("58dade241cb167f920eb090e1deed5dc")
        for question in dataset["questions"]:
            logger.info("Question:")
            logger.info("\t\t" + question)
            sql = pdsql.text2sql(question, ingestion)
            logger.info("Generated SQL:")
            logger.info("\t\t" + sql.replace("\n", " "))
            result = pd.read_sql(sql, db)
            logger.info("Result (5 sample rows):")
            if result.shape[0] > 5:
                logger.info("\t\t" + json.dumps(result.sample(5).to_json()) + "\n")
            else:
                logger.info("\t\t" + json.dumps(result.to_json()) + "\n")
