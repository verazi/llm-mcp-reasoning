import requests
from requests.auth import HTTPBasicAuth

API_KEY = "6c5e8a55ce6344cdb64bff8bbfe1a9d7"

def get_jwt(api_key: str) -> str:
    """Authenticate and return JWT token"""
    url = "https://api.aio.eresearch.unimelb.edu.au/login"
    res = requests.post(url, auth=HTTPBasicAuth('apikey', api_key))
    res.raise_for_status()
    return res.text.strip()

def test_version(jwt: str):
    """Call /version endpoint with JWT"""
    url = "https://api.aio.eresearch.unimelb.edu.au/version"
    headers = {"Authorization": f"Bearer {jwt}"}
    res = requests.get(url, headers=headers)
    res.raise_for_status()
    return res.text.strip()

if __name__ == "__main__":
    try:
        jwt = get_jwt(API_KEY)
        print("JWT success：", jwt[:50] + "...")

        version = test_version(jwt)
        print("API /version replies：", version)

    except Exception as e:
        print("Test fails:", e)
