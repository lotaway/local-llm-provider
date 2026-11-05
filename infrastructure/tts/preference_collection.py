from constants import MONGO_DB_NAME, MONGO_URI, TTS_MONGO_COLLECTION_PROFILES
from pymongo import MongoClient


def build_preference_collection(database: str = MONGO_DB_NAME):
    return MongoClient(MONGO_URI)[database][TTS_MONGO_COLLECTION_PROFILES]
