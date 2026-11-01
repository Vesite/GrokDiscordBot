A discord bot that acts like Grok, using Grok AI API. Designed for personal discord group chats, sometimes giving explicit or rude answers.

Requires a "keys.json" file with "discord_bot_token" and "grok_key" variables. This is not a part of the repository. It can optionally have a "xai_management_key" variable the /balance command.



### Features

Image generation, image editing and video generation, can be used by simply asking Grok in chat, or using the /generate commands

Ability to summarize the chat and search old messages, Grok decides to use search tools when needed.



### Optional Personalization

The folder "random_messages" can contain any number of files with the format "random_message_n". The bot has a 1% chance to respond with each of these messages. The file format can be .txt for text, an image (.png, .jpg, .jpeg, .gif, .webp) or a video (.mp4, .mov, .webm) under 10 MB. Files with the same name are sent together as one message




Using xAI - [x.ai/api](https://x.ai/api)
