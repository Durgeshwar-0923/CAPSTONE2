from sqlalchemy import create_engine
import pandas as pd
from src.config.config import DatabaseConfig
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class DataLoader:
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.engine = self._create_db_engine()
    
    def _create_db_engine(self):
        connection_string = (
            f"postgresql://{self.config.user}:{self.config.password}@"
            f"{self.config.host}:{self.config.port}/{self.config.database}"
        )
        return create_engine(connection_string)
    
    def load_data(self) -> pd.DataFrame:
        logger.info("Loading data from PostgreSQL")
        query = f"SELECT * FROM {self.config.table_name}"
        try:
            df = pd.read_sql(query, self.engine)
            logger.info(f"Successfully loaded {len(df)} records")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
