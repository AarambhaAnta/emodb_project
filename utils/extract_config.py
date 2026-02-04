import yaml

def get_config(config_path):
    with open(config_path, 'r') as config_file:
        config_data = yaml.safe_load(config_file)
    return config_data