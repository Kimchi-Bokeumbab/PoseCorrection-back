from app.database import SessionLocal
from app import models

db = SessionLocal()

for user in db.query(models.User).all():
    print(user.id, user.username, user.email)

db.close()

# 활동로그, 프로필 필드, 로그인 시도 제한