"""
Reads and validates data/config.json.
"""


import json
import os

def load_config(config_path: str = "data/config.json") -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    if "works" not in config:
        raise ValueError("Config must contain a 'works' key")
        
    project_root = os.getcwd()
    
    for work in config["works"]:
        if "id" not in work:
            raise ValueError("Each work entry must have an 'id'")
        if "resources" not in work:
            raise ValueError(f"Work {work['id']} must have a 'resources' key")
            
        for resource in work["resources"]:
            for field in ["lang", "format", "path"]:
                if field not in resource:
                    raise ValueError(f"Resource in work {work['id']} is missing required field: {field}")
            
            # Resolve relative paths to absolute paths from project root
            raw_path = resource["path"]
            if not os.path.isabs(raw_path):
                resource["path"] = os.path.abspath(os.path.join(project_root, raw_path))
                
    return config


def get_work(config: dict, work_id: str) -> dict:
    for work in config.get("works", []):
        if work.get("id") == work_id:
            return work
    raise ValueError(f"Work with ID '{work_id}' not found in config.")

