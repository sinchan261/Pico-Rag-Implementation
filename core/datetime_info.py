from datetime import datetime

def get_time():
    now = datetime.now()
    return now.strftime("%I:%M %p")

def get_date():
    now = datetime.now()
    return now.strftime("%d-%m-%Y")

def get_day():
    now = datetime.now()
    return now.strftime("%A")

def get_day_and_date():
    now = datetime.now()
    return now.strftime("%A, %d %B %Y")  
    # Example: Sunday, 25 August 2025
