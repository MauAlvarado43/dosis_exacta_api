import smtplib, ssl
from env import load_env

def send_email(subject, body, target):

    email_variables = load_env()

    smtp_server = email_variables["SMTP_SERVER"]
    port = email_variables["SMTP_PORT"]
    sender_email = email_variables["SMTP_EMAIL"]
    password = email_variables["SMTP_PASSWORD"]

    context = ssl.create_default_context()

    try:
        server = smtplib.SMTP(smtp_server,port)
        server.ehlo()
        server.starttls(context=context)
        server.ehlo() 
        server.login(sender_email, password)
        message = f"Subject: {subject}\n\n{body}"
        server.sendmail(sender_email, target, message)
    except Exception as e:
        print(e)
    finally:
        server.quit() 