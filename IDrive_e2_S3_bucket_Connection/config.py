from dotenv import load_dotenv
import os

load_dotenv()

S3_ENDPOINT = os.getenv("S3_ENDPOINT")
DATA_PATH = os.getenv("DATA_PATH")
