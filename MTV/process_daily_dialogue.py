import os
import json
import re
from typing import List

# Mapping for dialog act numbers to labels
DIALOG_ACT_MAP = {
    1: "inform",
    2: "question",
    3: "directive",
    4: "commissive",
}


def _clean_text(text: str) -> str:
    """Basic cleanup: collapse whitespace and strip."""
    # Replace multiple whitespace / newlines with single space
    cleaned = re.sub(r"\s+", " ", text)  # collapse whitespace
    # Remove spaces before punctuation (e.g. "city ." → "city.")
    cleaned = re.sub(r"\s+([.,!?;:’'])", r"\1", cleaned)
    # Remove spaces after apostrophes inside words ("can ’ t" → "can’t")
    cleaned = re.sub(r"([’'])\s+", r"\1", cleaned)
    return cleaned.strip()


def _read_utterances(text_file: str) -> List[str]:
    """Read all utterances from the dialogues_*.txt file.

    The DailyDialog raw text files contain multiple conversations. Each utterance
    ends with the token ``__eou__``.  Conversations may be broken across lines for
    readability, so we first read the *whole* file, collapse newlines to spaces,
    and then split on the utterance delimiter.
    """
    with open(text_file, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # New-lines are purely for formatting, not logical boundaries → replace with
    # spaces so we can split reliably on the end-of-utterance marker.
    raw_text = raw_text.replace("\n", " ")

    # Split on the delimiter and remove empty segments.
    utterances = [
        _clean_text(u) for u in raw_text.split("__eou__") if _clean_text(u)
    ]
    return utterances


def _read_act_lines(act_file: str) -> List[List[int]]:
    """Read dialog-act annotations – return list per conversation."""
    conversations_acts: List[List[int]] = []
    with open(act_file, "r", encoding="utf-8") as f:
        for line in f:
            acts = [int(tok) for tok in line.strip().split() if tok]
            if acts:
                conversations_acts.append(acts)
    return conversations_acts


def process_split(split_dir: str) -> List[dict]:
    """Convert a DailyDialog split directory into list of JSON examples."""
    # File names follow the pattern dialogues_<split>.txt, dialogues_act_<split>.txt
    split_name = os.path.basename(split_dir)
    text_file = os.path.join(split_dir, f"dialogues_{split_name}.txt")
    act_file = os.path.join(split_dir, f"dialogues_act_{split_name}.txt")

    if not (os.path.isfile(text_file) and os.path.isfile(act_file)):
        raise FileNotFoundError(f"Missing files in {split_dir}")

    utterances = _read_utterances(text_file)
    acts_per_conv = _read_act_lines(act_file)

    # Iterate through conversations, building examples
    data = []
    idx_pointer = 0  # pointer into utterance list
    conv_id = 0

    for acts in acts_per_conv:
        n_utts = len(acts)
        conv_utts = utterances[idx_pointer : idx_pointer + n_utts]

        # Sanity check
        if len(conv_utts) != n_utts:
            raise ValueError(
                f"Utterance count mismatch in {split_name} at conversation {conv_id}: "
                f"expected {n_utts}, got {len(conv_utts)}"
            )

        # Build examples for each utterance within this conversation
        history_lines: List[str] = []
        for local_idx, (utt_text, act_num) in enumerate(zip(conv_utts, acts)):
            caller = "A" if local_idx % 2 == 0 else "B"
            example = {
                "text": "\n".join(history_lines),  # context so far
                "response": utt_text,  # next turn to be generated
                "dialog_act": DIALOG_ACT_MAP.get(act_num, str(act_num)),
                "utterance_id": f"{conv_id}_{local_idx}",
                "caller": caller,
            }
            data.append(example)

            # Append the current utterance to history for the next step
            history_lines.append(f"{caller}: {utt_text}")

        # Move pointer and conversation counter
        idx_pointer += n_utts
        conv_id += 1

    # Extra sanity: we should have consumed *all* utterances
    if idx_pointer != len(utterances):
        raise ValueError(
            f"Unconsumed utterances: expected pointer {len(utterances)}, got {idx_pointer}"
        )

    return data


def save_jsonl(data: List[dict], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for obj in data:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    # Resolve paths relative to the location of this script so that the script
    # can be executed from any working directory.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_raw_dir = os.path.join(script_dir, "raw_data_daily_dialogue")
    # Output to MTV/data/dailydialog (relative to this script)
    output_dir = os.path.join(script_dir, "data", "dailydialog")
    split_dirs = [
        os.path.join(base_raw_dir, sub) for sub in ["train", "validation", "test"]
    ]

    for split_dir in split_dirs:
        split_name = os.path.basename(split_dir)
        print(f"Processing {split_name} split …")
        processed = process_split(split_dir)
        save_jsonl(processed, os.path.join(output_dir, f"{split_name}.json"))
        print(f"  → Saved {len(processed)} examples to {split_name}.json") 