import smtplib, ssl

def load_env_email():

    variables = {}

    with open(".env") as f:
        for line in f:
            key, value = line.split("=")
            variables[key] = value.strip()

    return variables

def send_email(subject, body, target):

    email_variables = load_env_email()

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