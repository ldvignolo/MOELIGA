import json
import telegram

# pip install python-telegram-bot

def notify(message):
    try:        
        with open('telegram.json', 'r') as keys_file:
            k = json.load(keys_file)
            token = k['telegram_token']
            chat_id = k['telegram_chat_id']
        bot = telegram.Bot(token=token)
        bot.sendMessage(chat_id=chat_id, text=message)
        
    except IOError:
        print("Notification config file not found.")   
    
  
