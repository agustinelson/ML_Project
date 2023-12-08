import json
import argparse

def load_args_from_json(json_file_path):
    with open(json_file_path, 'r') as file:
        args_dict = json.load(file)
    return argparse.Namespace(**args_dict)

# Ruta al archivo JSON
json_file_path = 'config.json'

# Cargar los argumentos desde el archivo JSON
args = load_args_from_json(json_file_path)

# Acceder a los valores de los argumentos
print(args.env_name)
print(args.lr)
# ... y así sucesivamente
