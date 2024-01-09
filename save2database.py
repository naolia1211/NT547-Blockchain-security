import os
from pymongo import MongoClient

# MongoDB setup
client = MongoClient('localhost', 27017)  # Default MongoDB port
db = client.blockchain
collection = db.graph

def save_files_to_mongodb(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        with open(file_path, 'r') as file:
            content = file.read()
            document = {
                'filename': filename,
                'content': content
            }
            # Insert document into MongoDB
            collection.insert_one(document)
            print(f"Saved {filename} to MongoDB.")

# Directory containing the files to save
directory = 'C:\\Users\\hao30\\Documents\\GitHub\\NT547-Blockchain-security\\New folder'

# Save files to MongoDB
save_files_to_mongodb(directory)
