import pandas as pd
from pymongo import MongoClient

# MongoDB connection settings
mongo_client = MongoClient("mongodb+srv://revankumar:revankumar@cluster0.pmcz5li.mongodb.net/")
db = mongo_client["Adult_census"]
collection = db["collection"]

# Read CSV file using pandas
csv_file_path = "notebooks/adult.csv"
data = pd.read_csv(csv_file_path)

# Convert the CSV data to a list of dictionaries
data_dict_list = data.to_dict(orient="records")

# Insert data into MongoDB collection
collection.insert_many(data_dict_list)

# Close MongoDB connection
mongo_client.close()
