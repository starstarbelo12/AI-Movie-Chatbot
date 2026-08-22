"""
audit_intent_keywords.py (auto-apply version)

Run this any time you add or edit patterns in intents.json.

What it does automatically:
  - Finds words used in intents.json that are missing from intent_only_keywords
  - Checks each missing word against your real movie titles
  - AUTOMATICALLY adds the safe ones (not found in any title) directly into
    text_processing.py for you
  - Leaves the risky ones (found in real titles) printed on screen only --
    those need your judgment, same call we made with "american"/"british"/
    "japanese" earlier. It will NOT add these without you deciding.

Usage (from the project root, same folder as app.py):
    python audit_intent_keywords.py

A backup of text_processing.py is saved as text_processing.py.bak before
any changes are written, in case you want to revert.
"""
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from chatbot.text_processing import intent_only_keywords, requirement_words
from chatbot.models import df_cleaned

INTENTS_PATH = Path(__file__).parent / "config" / "intents.json"
TEXT_PROCESSING_PATH = Path(__file__).parent / "src" / "chatbot" / "text_processing.py"

with open(INTENTS_PATH) as f:
    intents = json.load(f)

# Collect every word used across every pattern in intents.json
pattern_words = set()
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        pattern_words.update(re.findall(r"[a-z']+", pattern.lower()))

missing = sorted(pattern_words - requirement_words)

print(f"Total distinct words across all patterns: {len(pattern_words)}")
print(f"Words missing from requirement_words: {len(missing)}\n")

if not missing:
    print("Nothing missing -- no changes needed.")
    sys.exit(0)

all_titles_lower = df_cleaned["title"].astype(str).str.lower().tolist()

def example_titles_containing(word, limit=3):
    hits = []
    for title in all_titles_lower:
        if re.search(rf"\b{re.escape(word)}\b", title):
            hits.append(title)
            if len(hits) >= limit:
                break
    return hits

safe, risky = [], []
for word in missing:
    examples = example_titles_containing(word)
    if examples:
        risky.append((word, examples))
    else:
        safe.append(word)

# ------------------------------------------------------------------
# AUTO-APPLY the safe words into text_processing.py
# ------------------------------------------------------------------
if safe:
    source = TEXT_PROCESSING_PATH.read_text()

    match = re.search(
        r"(intent_only_keywords = \{)(.*?)(\n\})",
        source,
        flags=re.DOTALL,
    )
    if not match:
        print("Could not locate intent_only_keywords set in text_processing.py "
              "-- no changes made. Add these words manually:")
        print(", ".join(f'"{w}"' for w in safe))
    else:
        already_present = set(intent_only_keywords)
        new_words = [w for w in safe if w not in already_present]

        if not new_words:
            print("All safe words are already present -- nothing to add.")
        else:
            insertion = (
                "\n\n    # --- auto-added by audit_intent_keywords.py ---\n    "
                + ", ".join(f'"{w}"' for w in new_words) + ","
            )
            new_source = (
                source[:match.end(2)]
                + insertion
                + source[match.end(2):]
            )

            shutil.copy(TEXT_PROCESSING_PATH, TEXT_PROCESSING_PATH.with_suffix(".py.bak"))
            TEXT_PROCESSING_PATH.write_text(new_source)

            print(f"APPLIED: added {len(new_words)} safe words directly into "
                  f"{TEXT_PROCESSING_PATH.name}:")
            print(", ".join(f'"{w}"' for w in new_words))
            print(f"(backup saved as {TEXT_PROCESSING_PATH.name}.bak)")
else:
    print("No safe words to auto-add.")

# ------------------------------------------------------------------
# Print risky words for manual review only -- never auto-added
# ------------------------------------------------------------------
if risky:
    print(f"\nNOT added ({len(risky)}) -- these appear inside real movie titles. "
          f"Review manually and add yourself if the example titles are obscure:\n")
    for word, examples in risky:
        print(f'  "{word}"  -- e.g. {examples}')
