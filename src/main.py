from data.data import load_all_datasets

if __name__ == "__main__":
    print("Getting data")
    all_data = load_all_datasets(streaming=True)
    print("Dataset mapping initialization complete.")
