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
        auth: AuthConfig,
        s3_config: S3Config,
        env: str,
        model: str
    ):
        self.auth = auth
        self.environment = env
        self.s3 = s3_config
        self.model = model


config = AppConfig(
    auth=AuthConfig(
        secret=EnvOrThrow("SERVER_SECRET"), admin_token=EnvOrThrow("ADMIN_TOKEN")
    ),
    s3_config=S3Config(
        region=EnvOrThrow("AWS_REGION"),
        access_key=EnvOrThrow("AWS_ACCESS_KEY_ID"),
        secret_key=EnvOrThrow("AWS_SECRET_ACCESS_KEY_ID"),
        bucket=EnvOrThrow("AWS_BUCKET"),
        token=EnvOrThrow("AWS_TOKEN"),
    ),
    env=EnvOrThrow("ENVIRONMENT"),
    model= EnvOrThrow("EMBEDDING_MODEL")
    
)
