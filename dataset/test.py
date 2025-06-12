import pickle

def preview_pkl(path, num_samples=2):
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)

        print(f"Loaded {len(data)} items from {path}\n")

        tagged_count = sum(1 for item in data if item.get('tags'))
        print(f"Statistics: {tagged_count} items have tags ({tagged_count/len(data)*100:.1f}%)\n")

        for i, item in enumerate(data[:num_samples]):
            print(f"--- Sample #{i+1} ---")
            print(f"Prompt: {item.get('prompt', 'No prompt')}")
            print(f"Tags: {item.get('tags', 'No tags field')}")
            print()
    except FileNotFoundError:
        print(f"File not found: {path}")

if __name__ == "__main__":
  
    
    preview_pkl("extractors/data/pii_tagged.pkl")
    # preview_pkl("extractors/data/openchat_tagged_total.pkl")


# import pickle


# file_paths = [
#     "extractors/data/openchat_tagged_merge.pkl",
#     "extractors/data/openchat_tagged_parallel.pkl"
# ]

# valid_data = []
# invalid_data = []

# for path in file_paths:
#     with open(path, "rb") as f:
#         data = pickle.load(f)
#         if not isinstance(data, list):
#             print(f"file {path} is not list，skip。")
#             continue

#         for item in data:
#             if item.get("tags"):  # if it is valid tags
#                 valid_data.append(item)
#             else:
#                 invalid_data.append(item)

# # valid data
# with open("extractors/data/openchat_tagged_total.pkl", "wb") as f:
#     pickle.dump(valid_data, f)

# # invalid data
# with open("extractors/data/openchat_tagged_invalid.pkl", "wb") as f:
#     pickle.dump(invalid_data, f)


# print(f"   valid（with tag）: {len(valid_data)}")
# print(f"   invalid(without tag）: {len(invalid_data)}")



