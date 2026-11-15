from data.data import load_all_datasets


def main():
    print("Getting data")
    all_data = load_all_datasets(streaming=True)
    print("Dataset mapping initialization complete.")
