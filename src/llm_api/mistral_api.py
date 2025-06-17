import time

def call_mistral_api(client, prompt, model, temperature=0.0, call_delay=1.0):
    chat_response = client.chat.complete(
        model = model,
        temperature = temperature,
        messages = [
            {
                "role": "user",
                "content": prompt,
            },
        ]
    )
    time.sleep(call_delay)
    return chat_response.choices[0].message.content
        