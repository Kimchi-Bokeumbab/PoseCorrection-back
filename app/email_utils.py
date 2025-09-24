import smtplib
from email.message import EmailMessage

# def send_reset_email(to_email: str, token: str):
#     msg = EmailMessage()
#     msg['Subject'] = 'Password Reset'
#     msg['From'] = 'your_email@example.com'
#     msg['To'] = to_email
#     reset_link = f"http://localhost:8000/reset-password?token={token}"
#     msg.set_content(f"Click the link to reset your password: {reset_link}")

#     with smtplib.SMTP_SSL('smtp.example.com', 465) as server:
#         server.login('your_email@example.com', 'your_email_password')
#         server.send_message(msg)

# 테스트
def send_reset_email(to_email: str, token: str):
    reset_link = f"http://localhost:8000/reset-password?token={token}"
    print(f"[TEST EMAIL] To: {to_email}")
    print(f"[TEST EMAIL] Click the link to reset your password: {reset_link}")