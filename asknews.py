from asknews_sdk import AskNewsSDK

sdk = AskNewsSDK(
    client_id="1b85787f-67cc-4790-84a8-ac1211b2f0a6",
    client_secret="xzYT02L2z9TKmqq1.Zs4.KDVfP",
    scopes=["chat", "news", "stories"]
)

def chat_query():
    response = sdk.chat.get_chat_completions(
        messages=[
            {
                "role": "user",
                "content": "What's going on in the latest tech news?"
            }
        ],
        stream=False
    )


    print(response.choices[0].message.content)

chat_query()