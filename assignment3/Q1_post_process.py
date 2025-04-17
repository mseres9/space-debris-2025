import json

# Function to load JSON file as dictionary
def load_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Example usage for loading your JSON files
cdm_data = load_json_file('assignment3/output_Q1/cdm_results.json')
hie_data = load_json_file('assignment3/output_Q1/hie_results.json')
tca_data = load_json_file('assignment3/output_Q1/tca_results.json')

# You can now work with these dictionaries
print(cdm_data)
print(hie_data)
print(tca_data)

