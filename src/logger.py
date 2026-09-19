import logging
from src.config import ROOT

logs_path = ROOT / "logs"
logs_path.mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename=logs_path / "application.log",
                    format="[%(asctime)s] %(levelname)s %(message)s", level=logging.INFO)
