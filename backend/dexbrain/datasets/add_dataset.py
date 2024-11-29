from dexbrain.db_manager import DexBrainDB

def add_dataset(name, description, file_path):
    db = DexBrainDB()
    db.add_dataset(name, description, file_path)
    print(f"Dataset '{name}' added successfully.")
