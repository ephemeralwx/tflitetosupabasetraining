import os
import smtplib

from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

APP_PASSWORD = os.getenv('APP_PASSWORD')
from email.mime.text import MIMEText

subject = "test"
body = "test body"
sender = "kevinx8017@gmail.com"
recipients = [sender, "brianchen761@gmail.com"]
print(f"Email: {sender}")
print(f"App Password: {APP_PASSWORD}")


def send_email(subject, body, sender, recipients):
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = ", ".join(recipients)

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(sender, APP_PASSWORD)
        smtp.sendmail(sender, recipients, msg.as_string())
    print('Email sent successfully!')

send_email(subject, body, sender, recipients)