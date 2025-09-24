from sqlalchemy.orm import Session
from . import models, schemas
from passlib.context import CryptContext
from .auth import get_password_hash
from datetime import datetime, timedelta

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_user_by_username(db: Session, username: str):
    return db.query(models.User).filter(models.User.username == username).first()

def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()

def create_user(db: Session, user: schemas.UserCreate, is_admin: bool = False):
    hashed_password = get_password_hash(user.password)
    db_user = models.User(
        username=user.username,
        email=user.email,
        hashed_password=hashed_password,
        is_admin=is_admin
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def authenticate_user(db: Session, username: str, password: str):
    user = db.query(models.User).filter(models.User.username == username).first()
    if not user:
        return False
    from .auth import verify_password
    if not verify_password(password, user.hashed_password):
        return False
    return user

def record_failed_login(db: Session, user):
    user.failed_attempts += 1
    if user.failed_attempts >= 5:
        user.lock_until = datetime.utcnow() + timedelta(minutes=15)
    db.commit()
    db.refresh(user)
    return user

def reset_failed_login(db: Session, user):
    user.failed_attempts = 0
    user.lock_until = None
    db.commit()
    db.refresh(user)
    return user