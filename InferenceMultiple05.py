# goes through all files in files_inferencesorting/unsorted_files and copies them to the sorted_files subfolders (auto-generated).
import os
import shutil

from InferenceSingle04 import load_model, use_model

INFERENCE_DIR = "files_inferencesorting"
UNSORTED_DIR  = os.path.join(INFERENCE_DIR, "unsorted_files")
SORTED_DIR    = os.path.join(INFERENCE_DIR, "sorted_files")
AUDIO_FORMATS = ('.mp3', '.flac', '.wav')

def sort_files():
    # Collect all audio files from unsorted_files (non-recursive)
    candidates = [f for f in os.listdir(UNSORTED_DIR) if f.lower().endswith(AUDIO_FORMATS)]
    if not candidates:
        return
    model, classes, clip_length = load_model()
    print(f"Found {len(candidates)} file(s) to sort.\n")
    counts = {}
    for filename in sorted(candidates):
        filepath = os.path.join(UNSORTED_DIR, filename)
        results  = use_model(filepath, model, classes, clip_length)
        top_class, top_prob = results[0]
        # Create the destination subfolder (class) if it doesn't exist and copy the file
        dest_folder = os.path.join(SORTED_DIR, top_class)
        os.makedirs(dest_folder, exist_ok=True)
        shutil.copy2(filepath, os.path.join(dest_folder, filename))
        counts[top_class] = counts.get(top_class, 0) + 1
        bar = "█" * int(top_prob * 20)
        print(f"  {filename:<35}  ->  {top_class:<12}  {top_prob:>6.1%}  {bar}")
    print(f"\n{len(candidates)} file(s) sorted into {len(counts)} class folder(s).")

if __name__ == "__main__":
    sort_files()