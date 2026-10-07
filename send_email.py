"""
Email a results file. Used by the GitHub workflow after each run.

    python send_email.py --subject "NFL props" --html data/email_latest.html --attach a.csv b.csv
    python send_email.py --subject "Scoring" --text run.log --tail 120

Credentials come from environment variables (GitHub Actions secrets):
    EMAIL_USER          the sending Gmail address
    EMAIL_APP_PASSWORD  a Google "app password" (not your normal password)
    EMAIL_TO            where to send it (optional; defaults to EMAIL_USER)
Optional: EMAIL_SMTP_HOST / EMAIL_SMTP_PORT for a non-Gmail provider (SSL, default 465).

With no credentials set it prints a note and exits cleanly, so the workflow never fails
just because email isn't configured.
"""
import argparse
import html
import mimetypes
import os
import smtplib
import sys
from email.message import EmailMessage


def build_message(subject, sender, to, html_body=None, text_body=None, attachments=()):
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = subject, sender, to
    plain = text_body or "This email is best viewed in an HTML-capable mail app."
    msg.set_content(plain)
    if html_body:
        msg.add_alternative(html_body, subtype="html")
    for path in attachments:
        if not os.path.exists(path):
            print(f"Attachment not found, skipping: {path}")
            continue
        ctype, _ = mimetypes.guess_type(path)
        maintype, subtype = (ctype or "application/octet-stream").split("/", 1)
        with open(path, "rb") as f:
            msg.add_attachment(f.read(), maintype=maintype, subtype=subtype,
                               filename=os.path.basename(path))
    return msg


def send(msg) -> None:
    host = os.environ.get("EMAIL_SMTP_HOST", "smtp.gmail.com")
    port = int(os.environ.get("EMAIL_SMTP_PORT", "465"))
    with smtplib.SMTP_SSL(host, port, timeout=60) as s:
        s.login(os.environ["EMAIL_USER"], os.environ["EMAIL_APP_PASSWORD"])
        s.send_message(msg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--html", help="path to an HTML body")
    ap.add_argument("--text", help="path to a plain-text body (shown monospaced)")
    ap.add_argument("--tail", type=int, default=0, help="only send the last N lines of --text")
    ap.add_argument("--attach", nargs="*", default=[])
    a = ap.parse_args()

    user, pw = os.environ.get("EMAIL_USER"), os.environ.get("EMAIL_APP_PASSWORD")
    if not user or not pw:
        print("Email not configured (EMAIL_USER / EMAIL_APP_PASSWORD missing); skipping.")
        return 0
    to = os.environ.get("EMAIL_TO") or user

    html_body = text_body = None
    if a.html and os.path.exists(a.html):
        html_body = open(a.html, encoding="utf-8").read()
    elif a.text and os.path.exists(a.text):
        lines = open(a.text, encoding="utf-8", errors="replace").read().splitlines()
        if a.tail:
            lines = lines[-a.tail:]
        text_body = "\n".join(lines)
        html_body = ("<pre style=\"font-family:Menlo,Consolas,monospace;font-size:12px;"
                     f"white-space:pre-wrap\">{html.escape(text_body)}</pre>")
    else:
        print("Nothing to send (body file missing); skipping.")
        return 0

    send(build_message(a.subject, user, to, html_body, text_body, a.attach))
    print(f"Email sent to {to}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
