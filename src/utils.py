from datetime import datetime
import os
from typing import List, Dict

def get_timestamp():
    """
    Generate timestamp of format Y-M-D_H:M:S
    """
    return datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

def parse_model_list(models_string):
    return [model.strip() for model in models_string.split(',')]
