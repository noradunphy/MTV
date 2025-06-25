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

def json_to_excel(json_path: str, excel_path: str):
    """
    Load a JSON file with this structure:
      {
        "examples": [ { … }, { … }, … ],
        "summary": { … }
      }
    and write it to an Excel workbook with two sheets:
      - "Examples" (one row per example)
      - "Summary"  (single row with summary stats)
    """
    # 1) Load JSON
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 2) Flatten 'examples' into a DataFrame
    df_examples = pd.json_normalize(data["examples"])
    
    # Define the most important columns for quick viewing
    meta_cols = [
        "example_id", "current_dialogue", "target_output",
        "clean_output", "intervention_output",
        "target_dialogue_act", "clean_dialogue_act", "intervention_dialogue_act",
        "clean_correct", "intervention_correct",
        "clean_perplexity", "intervention_perplexity", "perplexity_difference"
    ]
    
    # Reorder columns to put meta_cols first, then any remaining columns
    available_meta_cols = [col for col in meta_cols if col in df_examples.columns]
    remaining_cols = [col for col in df_examples.columns if col not in meta_cols]
    df_examples = df_examples[available_meta_cols + remaining_cols]

    # 3) Turn 'summary' dict into a single-row DataFrame
    df_summary = pd.DataFrame([data["summary"]])

    # 4) Write to Excel with two sheets
    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        df_examples.to_excel(writer, sheet_name="Examples", index=False)
        df_summary.to_excel(writer, sheet_name="Summary", index=False)

    print(f"Saved Excel workbook: {excel_path}")

def json_to_csv(json_path: str, csv_path: str):
    """
    Load a JSON file and convert it to CSV format.
    """
    # 1) Load JSON
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 2) Flatten 'examples' into a DataFrame
    df_examples = pd.json_normalize(data["examples"])
    
    # Define the most important columns for quick viewing
    meta_cols = [
        "example_id", "current_dialogue", "target_output",
        "clean_output", "intervention_output",
        "target_dialogue_act", "clean_dialogue_act", "intervention_dialogue_act",
        "clean_correct", "intervention_correct",
        "clean_perplexity", "intervention_perplexity", "perplexity_difference"
    ]
    
    # Reorder columns to put meta_cols first, then any remaining columns
    available_meta_cols = [col for col in meta_cols if col in df_examples.columns]
    remaining_cols = [col for col in df_examples.columns if col not in meta_cols]
    df_examples = df_examples[available_meta_cols + remaining_cols]

    # 3) Save as CSV
    df_examples.to_csv(csv_path, index=False)
    print(f"Saved CSV file: {csv_path}")

def process_all_eval_results():
    """
    Process all eval_results_text_*.json files in the current directory.
    Creates both CSV and Excel versions for each file.
    """
    # Find all eval_results_text files
    pattern = "eval_results_text_*.json"
    json_files = glob.glob(pattern)
    
    if not json_files:
        print("No eval_results_text_*.json files found in current directory.")
        return
    
    print(f"Found {len(json_files)} eval_results_text files:")
    for file in json_files:
        print(f"  - {file}")
    
    print("\nProcessing files...")
    
    for json_file in json_files:
        # Create output filenames
        base_name = json_file.replace('.json', '')
        excel_file = f"{base_name}.xlsx"
        csv_file = f"{base_name}.csv"
        
        try:
            # Convert to Excel
            json_to_excel(json_file, excel_file)
            
            # Convert to CSV
            # json_to_csv(json_file, csv_file)
            
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")
    
    print("\nProcessing complete!")

# Example usage for a single file:
if __name__ == "__main__":
    # Process all eval_results_text files
    process_all_eval_results()
    
    # Uncomment below for single file processing:
    # json_to_excel(
    #     json_path="eval_results_text_swda_sv_maxlen100_resume.json",
    #     excel_path="eval_results_text_swda_sv_maxlen100_resume.xlsx"
    # )