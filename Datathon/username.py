import requests

# Twitter API v2 endpoint URL
username = "heybanco"
url = f"https://api.twitter.com/2/users/by/username/{username}"

# Request headers with authentication
headers = {
    "Authorization": "Bearer AAAAAAAAAAAAAAAAAAAAAJeQtgEAAAAAZsamRW7fDlor%2FttRLwCKGde0rL8%3D09RhmNUgunryu9HV7iSJzh8brgTJqD0a4aFhYpj8OmSMTl5jlu"
}

# Make the request
response = requests.get(url, headers=headers)

# Process the response
if response.status_code == 200:
    data = response.json()
    user_id = data['data']['id']
    print("User ID:", user_id)
else:
    print("Error:", response.status_code)
    print(response.text)
