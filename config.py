from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    root_path: Path = Path(__file__).parent
