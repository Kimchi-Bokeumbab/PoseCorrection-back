from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
from . import models, schemas, crud, database, auth
from typing import List
from .auth import get_password_hash
from .schemas import UserUpdate
from .database import get_db
from .email_utils import send_reset_email
from datetime import datetime, timedelta

app = FastAPI()

database.Base.metadata.create_all(bind=database.engine)

def init_admin():
    db = database.SessionLocal()
    try:
        admin_user = crud.get_user_by_username(db, "admin")
        if not admin_user:
            crud.create_user(db, schemas.UserCreate(
                username="admin",
                email="admin@test.com",
                password="admin123"
            ),
            is_admin=True)
    finally:
        db.close()

init_admin()

def get_current_user(token: str = Depends(auth.oauth2_scheme), db: Session = Depends(get_db)):
    payload = auth.verify_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token")

    username = payload.get("sub")
    if not username:
        raise HTTPException(status_code=401, detail="Invalid token payload")

    user = crud.get_user_by_username(db, username)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")
    return user

def get_current_admin(current_user: models.User = Depends(get_current_user)):
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin privileges required")
    return current_user

def admin_required(current_user: models.User = Depends(get_current_user)):
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="관리자 권한이 필요합니다."
        )
    return current_user

# 모든 사용자 조회 (관리자 전용)
@app.get("/users", response_model=List[schemas.UserOut], dependencies=[Depends(admin_required)])
def get_users(db: Session = Depends(get_db)):
    return db.query(models.User).all()

# 특정 사용자 조회 (관리자 전용)
@app.get("/users/{user_id}", response_model=schemas.UserOut, dependencies=[Depends(admin_required)])
def get_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

# 사용자 삭제 (관리자 전용)
@app.delete("/users/{user_id}", status_code=204, dependencies=[Depends(admin_required)])
def delete_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    db.delete(user)
    db.commit()
    return

# 사용자 수정 (관리자 전용)
@app.put("/users/{user_id}", response_model=schemas.UserOut, dependencies=[Depends(admin_required)])
def update_user(user_id: int, user_update: schemas.UserUpdate, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="user not found")
    if user_update.email:
        user.email = user_update.email
    if user_update.password:
        from .auth import get_password_hash
        user.hashed_password = get_password_hash(user_update.password)
    db.commit()
    db.refresh(user)
    return user

@app.get("/")
def read_root():
    return {"message": "Hello FastAPI"}

@app.post("/signup", response_model=schemas.UserOut)
def signup(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = crud.get_user_by_username(db, user.username)
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    return crud.create_user(db, user)

MAX_FAILED_ATTEMPTS = 5
LOCK_DURATION_MINUTES = 15

@app.post("/login", response_model=schemas.Token)
def login(login_req: schemas.LoginRequest, db: Session = Depends(get_db)):
    user = crud.get_user_by_username(db, login_req.username)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    if user.lock_until and user.lock_until > datetime.utcnow():
        raise HTTPException(status_code=403, detail="Account locked. Try again later.")
    
    if not auth.verify_password(login_req.password, user.hashed_password):
        crud.record_failed_login(db, user)
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    crud.reset_failed_login(db, user)

    access_token = auth.create_access_token(data={"sub": user.username})
    refresh_token = auth.create_refresh_token({"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/token/refresh")
def refresh_token(refresh_token: str):
    payload = auth.verify_token(refresh_token, token_type="refresh")
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

@app.get("/admin/users")
def read_users(db: Session = Depends(get_db), current_admin: models.User = Depends(get_current_admin)):
    users = db.query(models.User).all()
    return users

@app.patch("/users/me", response_model=schemas.UserOut)
def update_me(update: schemas.UserUpdate, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    if update.password:
        current_user.hashed_password = get_password_hash(update.password)
    if update.email:
        current_user.email = update.email
    db.commit()
    db.refresh(current_user)
    return current_user

@app.post("/password-reset-request")
def password_reset_request(
    request: schemas.PasswordResetRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    user = crud.get_user_by_email(db, request.email)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    token = auth.create_reset_token(user.email)

    # background_tasks.add_task(send_reset_email, user.email, token)

    # return {"msg": "Password reset email sent."}

    # 테스트
    return {"msg": "Password reset token created.", "token": token}

@app.post("/password-reset-confirm")
def password_reset_confirm(
    request: schemas.PasswordResetConfirm,
    db: Session = Depends(get_db)
):
    email = auth.verify_reset_token(request.token)
    if not email:
        raise HTTPException(status_code=400, detail="Invalid or expired token")
    
    user = crud.get_user_by_email(db, email)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    user.hashed_password = get_password_hash(request.new_password)
    db.commit()

    return {"msg": "Password successfully reset"}
