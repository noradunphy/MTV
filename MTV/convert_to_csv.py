# import json, pandas as pd

# json_path = "recomputed_perplexities_text_swda_b.json"   # your file
# csv_path  = "recomputed_perplexities_text_swda_b.csv"    # output

# # 1) Load the JSON
# with open(json_path) as f:
#     data = json.load(f)

# # 2) Flatten the “examples” list
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
    meta_cols = [
    "current_dialogue", "target_output",
    "clean_output", "intervention_output",
    "target_dialogue_act", "clean_dialogue_act", "intervention_dialogue_act",
    "clean_perplexity", "intervention_perplexity", "perplexity_difference"
    ]
    df_examples = df_examples[meta_cols + [c for c in df_examples.columns if c not in meta_cols]]

    # 3) Turn 'summary' dict into a single-row DataFrame
    df_summary = pd.DataFrame([data["summary"]])

    # 4) Write to Excel with two sheets
    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        df_examples.to_excel(writer, sheet_name="Examples", index=False)
        df_summary.to_excel(writer, sheet_name="Summary", index=False)

    print(f"Saved Excel workbook: {excel_path}")

# Example usage:
if __name__ == "__main__":
    json_to_excel(
        json_path="recomputed_perplexities_text_swda_sv.json",
        excel_path="recomputed_perplexities_text_swda_sv.xlsx"
    )