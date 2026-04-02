import os
from dotenv import load_dotenv

load_dotenv()

def EnvOrThrow(key: str):
    var = os.getenv(key)
    if not var:
        raise KeyError(f"Missing environment variable {key}")
    return var


class AuthConfig:
    def __init__(self, secret: str, admin_token: str, jwt_expiry: int = 3600):
        self.secret = secret
        self.jwt_expiry = jwt_expiry
        self.admin_token = admin_token


class S3Config:
    def __init__(
        self, region: str, access_key: str, secret_key: str, token: str, bucket: str
    ):
        self.region = region
        self.key = access_key
        self.secret_key = secret_key
        self.token = token
        self.bucket = bucket


class AppConfig:
    def __init__(
        self,
        env: str,
        model: str,
        med_data_url:str
    ):
        self.environment = env
        self.model = model
        self.med_data_url = med_data_url


config = AppConfig(
    env=EnvOrThrow("ENVIRONMENT"),
    model= EnvOrThrow("EMBEDDING_MODEL"),
    med_data_url=EnvOrThrow("MEDICINE_DATA_URL")
    
)
