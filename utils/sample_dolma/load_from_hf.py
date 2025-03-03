from datasets import get_dataset_split_names, load_dataset
# ------------------- find all splits, for no only train -------------------
splits = get_dataset_split_names("allenai/dolma")
print(splits)

# ------------------- load Full dataset -------------------
# import os
# from datasets import load_dataset

# os.environ["DATA_DIR"] = "/home/eisape/projects/diversify_lm_output/dolma/data"
# dataset = load_dataset("allenai/dolma", split="train")

# ------------------- load One dataset -------------------

dataset = load_dataset("allenai/dolma", split="train", streaming=True)
first_example = next(iter(dataset)) 
print(first_example)


