import os
import re
import yaml
from pathlib import Path


def read_yaml_file(file_path: Path) -> dict:
    try:
        with open(file_path, "r") as stream:
            return yaml.safe_load(stream)
    except FileNotFoundError as e:
        print(f"Error: Could not find {file_path}: {e}")
        return {}
    except yaml.YAMLError as exc:
        print(f"Error: Could not read {exc}: {e}")
        return {}
