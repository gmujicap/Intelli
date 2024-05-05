import requests
import csv

# Twitter API v2 endpoint URL
url = "https://api.twitter.com/2/tweets/search/recent"

# Request headers with authentication
headers = {
    "Authorization": "Bearer AAAAAAAAAAAAAAAAAAAAAJeQtgEAAAAAA2N9tzy9mGzIzclaCd7k9GD2sm8%3DQk3iuNcS4TGIXWHXjYoqafXuuMt3dVsb8WjZwzlsoEQoBqdHnZ",
}

# Request parameters
params = {
    "query": "from:heybanco",
    "max_results": 100,  # Maximum number of tweets to retrieve per request
    "tweet.fields": "id,text,created_at",  # Additional fields you want to include
}

# Make the request
response = requests.get(url, headers=headers, params=params)

# Print the response content
print(response.text)

# Process the response
if response.status_code == 200:
    data = response.json()
    # Process and save tweets to CSV
    with open('user_timeline.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Tweet ID', 'Text', 'Created At']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for tweet in data['data']:
            writer.writerow({'Tweet ID': tweet['id'], 'Text': tweet['text'], 'Created At': tweet['created_at']})
else:
    print("Error:", response.status_code)
    print(response.text)
