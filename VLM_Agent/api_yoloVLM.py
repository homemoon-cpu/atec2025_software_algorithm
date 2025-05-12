
# import requests
# import ollama
# import time


# def call_api_vlm(prompt, base64_image=None):


#     ollama_client = ollama.Client()

#     start_time = time.time()
#     response = ollama_client.chat(model='gemma3:27b',
#     messages=[{
#         'role': 'system',
#         'content': prompt,
#         },
#         {
#             'role': 'user',
#             'content':"Robot's observation",
#             "images": [base64_image]
#         }
#     ]
#     )
#     end_time = time.time()
#     response_time = end_time - start_time
    

#     return response['message']['content']







# def call_api_llm(prompt):

#     input_messages = [
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": prompt}
#     ]

#     ollama_client = ollama.Client()
#     start_time = time.time()
#     response = ollama_client.chat(model='gemma3:27b',
#     messages=input_messages
#     )
#     end_time = time.time()
#     response_time = end_time - start_time

#     return response['message']['content']

from openai import OpenAI
import requests
import os



def call_api_vlm(prompt, base64_image=None):

    client = OpenAI(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=os.getenv("DASHSCOPE_API_KEY")
    )

    response = client.chat.completions.create(
        model='qwen2.5-vl-72b-instruct',
        # max_tokens=512,
        messages=[
            {"role": "user", "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }]
            },
        ],

    )

    return response.choices[0].message.content






def call_api_llm(prompt):

    input_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    url = "https://cloud.infini-ai.com/maas/qwen2.5-72b-instruct/nvidia/chat/completions"

    api_key = os.getenv("INFINI_API_KEY")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": 'qwen2.5-72b-instruct',
        "messages": input_messages,
        "temperature": 0.6,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    response_json = response.json()
    response.close()

    return response_json["choices"][0]["message"]["content"]










