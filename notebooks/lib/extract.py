import os
import re
import json

# Directory containing the Markdown files
directory = 'nba_api/docs/nba_api/stats/endpoints'

# Output file
output_file = 'filtered_datasets_with_pie.md'

# Regular expression to match JSON sections
json_pattern = re.compile(r'```json\n(.*?)\n```', re.DOTALL)

# Open the output file in write mode
with open(output_file, 'w', encoding='utf-8') as outfile:
    # Iterate through each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.md'):
            filepath = os.path.join(directory, filename)
            try:
                # Attempt to open the file with UTF-8 encoding
                with open(filepath, 'r', encoding='utf-8') as file:
                    content = file.read()
            except UnicodeDecodeError:
                # Fallback to a different encoding if UTF-8 fails
                with open(filepath, 'r', encoding='latin-1') as file:
                    content = file.read()
            # Find all JSON sections
            json_sections = json_pattern.findall(content)
            if json_sections:
                # Write the filename as a header
                outfile.write(f'# {filename}\n\n')
                # Process each JSON section
                for json_text in json_sections:
                    try:
                        json_data = json.loads(json_text)
                        data_sets = json_data.get("data_sets", {})
                        # Filter data_sets for lists containing "PIE"
                        filtered_data_sets = {
                            key: values for key, values in data_sets.items()
                            if "PIE" in values
                        }
                        # Only write if there are relevant datasets
                        if filtered_data_sets:
                            outfile.write(f'```json\n')
                            outfile.write(json.dumps({"data_sets": filtered_data_sets}, indent=4))
                            outfile.write(f'\n```\n\n')
                    except json.JSONDecodeError:
                        # Handle invalid JSON sections gracefully
                        continue
    print(f"Filtered data sets with 'PIE' saved to {output_file}")
