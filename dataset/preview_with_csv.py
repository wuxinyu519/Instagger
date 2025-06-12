import pickle
import csv
import os

def convert_pkl_to_csv(input_path, output_path):
   
    with open(input_path, "rb") as f:
        data = pickle.load(f)


   
    fieldnames = list(data[0].keys())

 
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
 
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
          
            for key in row:
                if isinstance(row[key], list):
                    row[key] = ", ".join(map(str, row[key]))
            writer.writerow(row)

    print(f"Converted {len(data)} rows from {input_path} to {output_path}")


if __name__ == "__main__":
    convert_pkl_to_csv("extractors/data/openchat_tagged_invalid.pkl", "extractors/csv/openchat_tagged_invalid.csv")
