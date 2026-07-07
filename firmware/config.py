from typing import Any

import tomlkit
from pathlib import Path

class Config:

    def __init__(self, config_file):
        self.configurations = self.load_configurations(config_file)

    def load_configurations(self, config_file):
        document = tomlkit.parse(Path(config_file).read_text()) if Path(config_file).exists() else tomlkit.document()
        return document

    def save_configurations(self, config_file):
        with open(config_file, "w") as f:
            tomlkit.dump(self.configurations, f)

    def getString(self, key: str, default="") -> str:
        return self.configurations.get(key, default)

    def getInt(self, key: str, default=0) -> int:
        return int(self.configurations.get(key, default))
    
    def getList(self, key: str, default=[]) -> list:
        return self.configurations.get(key, default)

    def getFloat(self, key: str, default=0.0) -> float:
        return float(self.configurations.get(key, default))

    def getBoolean(self, key: str, default=False) -> bool:
        return bool(self.configurations.get(key, default))
    
    def set(self, key: str, value: Any):
        self.configurations[key] = value