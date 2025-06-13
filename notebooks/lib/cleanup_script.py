import os
from yapf.yapflib.yapf_api import FormatCode

def remove_markdown_blocks_and_reformat(input_path, output_path):
    lines_to_keep = []
    skip_markdown = False

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            # If we see a markdown block start, skip until next code block
            if line.strip().startswith("# %% [markdown]"):
                skip_markdown = True
                continue
            if skip_markdown:
                if line.strip().startswith("# %%"):
                    skip_markdown = False  # Found next code block
                    lines_to_keep.append(line)
                continue
            # Otherwise, keep the line
            lines_to_keep.append(line)

    # Join lines into a single string
    code_str = "".join(lines_to_keep)

    # Reformat with YAPF using custom style settings
    formatted_code, _ = FormatCode(
        code_str,
        
        style_config={
            "based_on_style": "pep8",
            "column_limit": 400,         # Increase line length to allow fewer line breaks
            "COALESCE_BRACKETS": True,  # Attempt to coalesce bracket pairs
            # Attempt to reduce splits:
            "ALLOW_SPLIT_BEFORE_DICT_VALUE": False,
            "SPLIT_BEFORE_NAMED_ASSIGNS": False,
        }
    )

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(formatted_code)


if __name__ == "__main__":
    input_file = "notebooks/test.py"
    output_file = "notebooks/test_cleaned.py"
    
    remove_markdown_blocks_and_reformat(input_file, output_file)
    print(f"Cleaned file saved to {output_file}")
