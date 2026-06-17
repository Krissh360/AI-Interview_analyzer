import pandas as pd
import glob
import os

def clean_csv_files():
    # Find all CSV files in the current directory
    csv_files = glob.glob("*.csv")
    
    if not csv_files:
        print("No CSV files found in the current directory.")
        return

    # Create a directory to store the cleaned files
    output_dir = "cleaned_batches"
    os.makedirs(output_dir, exist_ok=True)

    for file in csv_files:
        print(f"Processing {file}...")
        try:
            # Read the CSV. Pandas automatically respects quotes holding newlines.
            df = pd.read_csv(file)
            
            # Check if the 'answer' column exists
            if 'answer' in df.columns:
                # Replace any carriage returns or newlines with a single space
                df['answer'] = df['answer'].str.replace(r'[\r\n]+', ' ', regex=True)
                
                # Save to the new folder
                output_path = os.path.join(output_dir, file)
                df.to_csv(output_path, index=False)
                print(f"  -> Successfully cleaned and saved to {output_path}")
            else:
                print(f"  -> Skipping: No 'answer' column found in {file}")
                
        except Exception as e:
            print(f"  -> Error processing {file}: {e}")

if __name__ == "__main__":
    clean_csv_files()