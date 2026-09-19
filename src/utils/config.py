import os
import yaml


class ConfigNode(dict):
    """Dictionary hỗ trợ truy cập dạng thuộc tính cfg.key."""
    def __init__(self, *args, **kwargs):
        super(ConfigNode, self).__init__(*args, **kwargs)
        for k, v in self.items():
            if isinstance(v, dict):
                self[k] = ConfigNode(v)
            elif isinstance(v, list):
                self[k] = [ConfigNode(i) if isinstance(i, dict) else i for i in v]

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"Không tìm thấy khóa cấu hình: '{key}'")

    def __setattr__(self, key, value):
        self[key] = value


def load_config(config_path: str) -> ConfigNode:
    """Đọc file cấu hình YAML."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Không tìm thấy tệp cấu hình tại: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        raw_dict = yaml.safe_load(f)
    
    return ConfigNode(raw_dict)
