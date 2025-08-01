import json
import requests
import time

# Configuration
url = 'http://localhost:5000/score'
headers = {
    'Content-Type': 'application/json',
    'X-Forwarded-For': f"1.2.3.1"  # simulasi 40 IP berbeda
}

data = {
    'answer': "asdfasdfasdf",
    'reference': "gatau"
}

success_count = 0
fail_count = 0

start_time = time.time()

for _ in range(40):
    try:
        response = requests.post(url, data=json.dumps(data), headers=headers, timeout=10)
        if response.status_code == 200:
            success_count += 1
        else:
            fail_count += 1
    except Exception:
        fail_count += 1

end_time = time.time()
total_time = end_time - start_time

(success_count, fail_count, total_time)
