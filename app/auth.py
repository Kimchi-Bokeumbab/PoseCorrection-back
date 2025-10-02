from datetime import datetime, timedelta

from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext


SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: int | None = None) -> str:
    to_encode = data.copy()
    if expires_delta is None:
        expires_delta = ACCESS_TOKEN_EXPIRE_MINUTES
    expire = datetime.utcnow() + timedelta(minutes=expires_delta)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(data: dict, expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS))
    to_encode.update({"exp": expire, "type": "refresh"})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str, token_type: str | None = None) -> dict | None:
    """Decode and validate a JWT.

    Returns the decoded payload when the token is valid, otherwise ``None``.
    When ``token_type`` is provided the decoded payload must contain the same
    ``type`` value.
    """

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None

    if token_type and payload.get("type") != token_type:
        return None

    return payload


RESET_SECRET_KEY = "your-reset-secret-key"
RESET_ALGORITHM = "HS256"
RESET_TOKEN_EXPIRE_MINUTES = 30


def create_reset_token(email: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=RESET_TOKEN_EXPIRE_MINUTES)
    to_encode = {"sub": email, "exp": expire}
    return jwt.encode(to_encode, RESET_SECRET_KEY, algorithm=RESET_ALGORITHM)


def verify_reset_token(token: str) -> str | None:
    try:
        payload = jwt.decode(token, RESET_SECRET_KEY, algorithms=[RESET_ALGORITHM])
    except JWTError:
        return None

    email = payload.get("sub")
    if email is None:
        return None

    return email
