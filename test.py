import requests

API_BASE = "http://localhost:11434/v1"
# API_BASE = "https://api.apiyi.com/v1"
CHAT_API = API_BASE + "/chat/completions"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Bearer YOUR_ACCESS_TOKEN",  # Use your own access token here
}


def test_non_stream():
    print("=== Non-stream request ===")
    resp = requests.post(
        CHAT_API,
        headers=HEADERS,
        json={
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "Hello, introduce yourself"}],
            "stream": False,
        },
        stream=False,
    )
    print(resp.json())
    print("=== Non-stream request end ===\n")


def test_stream():
    print("=== Stream request ===")
    resp = requests.post(
        CHAT_API,
        headers=HEADERS,
        json={
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "Hello, introduce yourself"}],
            "stream": True,
        },
        stream=True,  # Key word argument
    )

    for line in resp.iter_lines(decode_unicode=True):
        if line:
            print(line)
    print("=== Stream request end ===")


if __name__ == "__main__":
    # test_non_stream()
    test_stream()
