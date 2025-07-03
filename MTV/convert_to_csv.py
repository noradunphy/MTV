# import json, pandas as pd

# json_path = "recomputed_perplexities_text_swda_b.json"   # your file
# csv_path  = "recomputed_perplexities_text_swda_b.csv"    # output

# # 1) Load the JSON
# with open(json_path) as f:
#     data = json.load(f)

# # 2) Flatten the "examples" list
# df = pd.json_normalize(data["examples"])

# # 4) Re-order the most important columns for quick viewing
# meta_cols = [
#     "current_dialogue", "target_output",
#     "clean_output", "intervention_output",
#     "target_dialogue_act", "clean_dialogue_act", "intervention_dialogue_act",
#     "clean_perplexity", "intervention_perplexity", "perplexity_difference"
# ]
# df = df[meta_cols + [c for c in df.columns if c not in meta_cols]]

# # 5) Save as comma-separated values
# df.to_csv(csv_path, index=False)


import json
import pandas as pd
import glob
import os

def extract_current_dialogue_and_options(input_text: str, current_dialogue: str) -> dict:
    """
    Extract the current dialogue's options from the input_text by finding the last example
    which matches the current_dialogue.
    """
    # Split input text into examples based on "Response:" marker
    examples = input_text.split("Response:")
    
    # Get the last example (current dialogue)
    current_example = examples[-1].strip()
    
    # Extract options from the current example
    options = []
    lines = current_example.split("\n")
    for line in lines:
        if "1)" in line:
            # Extract all 4 options from this line
            parts = line.split("1)")[1].split("2)")
            opt1 = parts[0].strip()
            parts = parts[1].split("3)")
            opt2 = parts[0].strip()
            parts = parts[1].split("4)")
            opt3 = parts[0].strip()
            opt4 = parts[1].strip()
            options = [opt1, opt2, opt3, opt4]
            break
    
    # Extract ICL examples
    icl_examples = examples[:-1]
    icl_text = "Response:".join(icl_examples).strip()
    
    return {
        "options": options,
        "icl_examples": icl_text
    }

def json_to_excel(json_path: str, excel_path: str):
    """
    Load a JSON file with MCQ format and write it to an Excel workbook with two sheets:
      - "Examples" (one row per example)
      - "Summary"  (single row with summary stats)
    """
    # 1) Load JSON
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 2) Flatten 'examples' into a DataFrame
    df_examples = pd.json_normalize(data["examples"])
    
    # 3) Process each example to extract options and ICL examples
    options_list = []
    icl_examples_list = []
    
    for _, row in df_examples.iterrows():
        extracted = extract_current_dialogue_and_options(row['input_text'], row['current_dialogue'])
        options_list.append(extracted['options'])
        icl_examples_list.append(extracted['icl_examples'])
    
    # Add new columns for options and ICL examples
    df_examples['option_1'] = [opts[0] if opts else None for opts in options_list]
    df_examples['option_2'] = [opts[1] if opts else None for opts in options_list]
    df_examples['option_3'] = [opts[2] if opts else None for opts in options_list]
    df_examples['option_4'] = [opts[3] if opts else None for opts in options_list]
    df_examples['icl_examples'] = icl_examples_list
    
    # Define the most important columns for quick viewing
    meta_cols = [
        "example_id", "current_dialogue", "next_caller",
        "option_1", "option_2", "option_3", "option_4",
        "target_output", "target_dialogue_act",
        "clean_dialogue_act", "intervention_dialogue_act",
        "icl_examples"
    ]
    
    # Reorder columns to put meta_cols first, then any remaining columns
    available_meta_cols = [col for col in meta_cols if col in df_examples.columns]
    remaining_cols = [col for col in df_examples.columns if col not in meta_cols]
    df_examples = df_examples[available_meta_cols + remaining_cols]

    # 4) Turn 'summary' dict into a single-row DataFrame
    df_summary = pd.DataFrame([data["summary"]])

    # 5) Write to Excel with two sheets
    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        df_examples.to_excel(writer, sheet_name="Examples", index=False)
        df_summary.to_excel(writer, sheet_name="Summary", index=False)

    print(f"Saved Excel workbook: {excel_path}")

def process_all_eval_results():
    """
    Process all eval_results_text_*_MCQ.json files in the current directory.
    Creates Excel versions for each file.
    """
    # Find all eval_results_text MCQ files
    pattern = "eval_results_text_*_MCQ.json"
    json_files = glob.glob(pattern)
    
    if not json_files:
        print("No eval_results_text_*_MCQ.json files found in current directory.")
        return
    
    print(f"Found {len(json_files)} MCQ eval results files:")
    for file in json_files:
        print(f"  - {file}")
    
    print("\nProcessing files...")
    
    for json_file in json_files:
        # Create output filename
        base_name = json_file.replace('.json', '')
        excel_file = f"{base_name}.xlsx"
        
        try:
            # Convert to Excel
            json_to_excel(json_file, excel_file)
            
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    # Process all MCQ eval results files
    process_all_eval_results()