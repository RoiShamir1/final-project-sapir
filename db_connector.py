from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import datetime

class MongoDBClient:
    def __init__(self, connection_string: str, db_name="drone_security_db", collection_name="alerts"):
        self.uri = connection_string
        self.db_name = db_name
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self.is_connected = False

    def connect(self):
        """מנסה להתחבר לשרת הענן"""
        print("☁️ Connecting to MongoDB Atlas...")
        try:
            # timeoutMS=5000 אומר שאם תוך 5 שניות אין חיבור - הוא מוותר
            self.client = MongoClient(self.uri, serverSelectionTimeoutMS=5000)
            
            # בדיקת חיבור מהירה (פינג)
            self.client.admin.command('ping')
            
            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]
            self.is_connected = True
            print("✅ Connected to Cloud Database successfully!")
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            print(f"⚠️ Cloud Connection Failed: {e}")
            print("   (System will continue working offline)")
            self.is_connected = False

    def insert_event(self, event_dict: dict):
        """שולח את האירוע לענן"""
        if not self.is_connected:
            return # אם אין חיבור, לא מנסים אפילו

        try:
            # MongoDB מוסיף אוטומטית שדה _id ייחודי
            result = self.collection.insert_one(event_dict)
            print(f"📡 Uploaded to Cloud. ID: {result.inserted_id}")
        except Exception as e:
            print(f"❌ Upload Failed: {e}")