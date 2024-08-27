import ast
import json
import requests
import time

url = "https://YOUR_MODAL_USERNAME_AND_APP_NAME-web-dev.modal.run" # replace with the result of `modal deploy modal_web_endpoint.py`
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer YOUR_MODAL_API_KEY", # replace with your Modal API Key
}

payload = {
    "prompts": ["What is the capital of France?"],
}

start_time = time.time()
response = requests.post(url, headers=headers, json=payload)
end_time = time.time()

if response.status_code == 200:
    response_list = ast.literal_eval(response.text)
    for i in response_list:
        print("=" * 50)
        print(i)
    
    # Add details about the number of answers, time taken, and average time per task
    print("\n" + "=" * 50)
    total_time = end_time - start_time
    num_tasks = len(response_list)
    print(f"Number of answers: {num_tasks}")
    print(f"Total time taken: {total_time:.2f} seconds")
    print(f"Average time per task: {total_time / num_tasks:.2f} seconds")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
