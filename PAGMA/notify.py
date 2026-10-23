"""Optional email notification. Credentials ONLY via environment variables."""
import os
import smtplib
from email.mime.text import MIMEText


def send(subject: str, body: str) -> bool:
    user, pwd, to = os.environ.get("SMTP_USER"), os.environ.get("SMTP_PASSWORD"), os.environ.get("NOTIFY_TO")
    if not (user and pwd and to):
        print("[notify] SMTP_USER / SMTP_PASSWORD / NOTIFY_TO undefined : e-mail ignored.")
        return False
    msg = MIMEText(body)
    msg["From"], msg["To"], msg["Subject"] = user, to, subject
    try:
        with smtplib.SMTP(os.environ.get("SMTP_HOST", "smtp.gmail.com"),
                          int(os.environ.get("SMTP_PORT", "587"))) as s:
            s.starttls()
            s.login(user, pwd)
            s.sendmail(user, [to], msg.as_string())
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[notify] fail: {e}")
        return False
