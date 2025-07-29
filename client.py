import json
import requests

data = {
    'answer': "asdfasdfasdf",
    'reference': "gatau"
}

url = 'http://localhost:5001/score'
headers = {'Content-Type': 'application/json'}  # Add this header
response = requests.post(url, data=json.dumps(data), headers=headers)
print(response.json())
