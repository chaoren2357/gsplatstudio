import os
import json
import shutil

def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)

def should_exclude(item_path, excluded_dirs, excluded_files):
    return item_path.name in excluded_dirs or item_path.name in excluded_files or item_path.name.startswith('.')

def copy_items(src_path, dest_path, excluded_dirs, excluded_files):
    for item in src_path.iterdir():
        if should_exclude(item, excluded_dirs, excluded_files):
            continue
        new_dest = dest_path / item.name
        if item.is_dir():
            new_dest.mkdir(exist_ok=True)
            copy_items(item, new_dest, excluded_dirs, excluded_files)
        else:
            shutil.copyfile(item, new_dest)

def load_json(json_file):
    with open(json_file, 'r') as file:
        return json.load(file)