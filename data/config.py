from typing import Any

import tomlkit
from pathlib import Path

class Config:

    def __init__(self, config_file):
        self.config_file = config_file
        self.configurations = self.load_configurations()

    def load_configurations(self):
        document = tomlkit.parse(Path(self.config_file).read_text()) if Path(self.config_file).exists() else tomlkit.document()
        return document

    def save_configurations(self):
        with open(self.config_file, "w") as f:
            tomlkit.dump(self.configurations, f)

    def getString(self, key: str, default="") -> str:
        if key not in self.configurations:
            self.configurations[key] = default

        return self.configurations.get(key, default)

    def getInt(self, key: str, default=0) -> int:
        if key not in self.configurations:
            self.configurations[key] = default

        return int(self.configurations.get(key, default))
    
    def getList(self, key: str, default=[]) -> list:
        if key not in self.configurations:
            self.configurations[key] = default

        return self.configurations.get(key, default)

    def getFloat(self, key: str, default=0.0) -> float:
        if key not in self.configurations:
            self.configurations[key] = default

        return float(self.configurations.get(key, default))

    def getBoolean(self, key: str, default=False) -> bool:
        if key not in self.configurations:
            self.configurations[key] = default
            
        return bool(self.configurations.get(key, default))
    
    def set(self, key: str, value: Any):
        self.configurations[key] = value