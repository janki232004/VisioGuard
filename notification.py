import os
import json

import requests


def pushbullet_noti(title, body):
    token = os.getenv("PUSHBULLET_TOKEN", "o.TAiuPBTxUXxynukN4r7iLZYVXMyjAWr2").strip()
    if not token:
        raise RuntimeError("Pushbullet token is missing. Set PUSHBULLET_TOKEN.")

    msg = {"type": "note", "title": title, "body": body}
    resp = requests.post(
        "https://api.pushbullet.com/v2/pushes",
        data=json.dumps(msg),
        headers={
            "Authorization": "Bearer " + token,
            "Content-Type": "application/json",
        },
        timeout=8,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Pushbullet error {resp.status_code}: {resp.text}")

    print("Pushbullet message sent")
