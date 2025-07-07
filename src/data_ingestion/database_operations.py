import pandas as pd
from sqlalchemy import create_engine
from src.config.config import DatabaseConfig
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def load_csv_to_postgres(csv_path: str = "data/raw/Software_Salaries.csv", table_name: str = None) -> None:
    config = DatabaseConfig()
    if table_name is None:
        table_name = config.table_name

    engine = create_engine(config.db_url)


    logger.info(f"📤 Loading CSV: {csv_path} → PostgreSQL table: {table_name}")
    df = pd.read_csv(csv_path)
    df.to_sql(table_name, engine, if_exists="replace", index=False)
    logger.info("✅ Data loaded into PostgreSQL successfully.")
