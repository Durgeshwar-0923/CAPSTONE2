from dataclasses import dataclass, field
from pathlib import Path
import os

@dataclass
class DatabaseConfig:
    host: str = os.getenv("DB_HOST", "localhost")
    port: int = int(os.getenv("DB_PORT", 5432))
    database: str = os.getenv("DB_NAME", "salary_db")
    user: str = os.getenv("DB_USER", "postgres")
    password: str = os.getenv("DB_PASS", "Minfy%40Durgesh")
    table_name: str = "salary_data"

    @property
    def db_url(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

@dataclass
class MLConfig:
    target_column: str = "total_compensation"
    random_state: int = 42
    test_size: float = 0.2
    models_to_train: list[str] = field(default_factory=lambda: ['rf', 'xgb', 'lgbm', 'catboost'])

@dataclass
class Paths:
    ROOT: Path = Path(__file__).parent.parent.parent
    DATA: Path = ROOT / "data"
    MODELS: Path = ROOT / "models"
    REPORTS: Path = ROOT / "reports"

    RAW_DATA: Path = DATA / "raw"
    PROCESSED_DATA: Path = DATA / "processed"
    DRIFT_REPORTS: Path = DATA / "drift_reports"

@dataclass
class MLOpsConfig:
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    airflow_dag_folder: Path = Paths.ROOT / "airflow" / "dags"
