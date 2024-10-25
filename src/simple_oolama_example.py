import ollama

# A simple prompt and response
response = ollama.chat(model='mistral', messages=[
    {
        'role': 'user',
        'content': 'Why is the sky blue?',
    },
])

# Print Response
print(response['message']['content'])

