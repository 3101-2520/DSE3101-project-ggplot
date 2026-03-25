import csv
import os
import re
from pathlib import Path

# Saves the CSV in the same directory as this file
current_dir = Path(__file__).parent
SUBSCRIBER_FILE = current_dir / "subscribers.csv"

def save_subscriber(email):
    email = email.strip().lower()
    if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        return False, "Invalid email format."

    file_exists = SUBSCRIBER_FILE.exists()
    
    if file_exists:
        with open(SUBSCRIBER_FILE, mode='r', newline='') as f:
            reader = csv.reader(f)
            # Check for duplicates
            if any(row and row[0] == email for row in reader):
                return False, "Already subscribed!"

    with open(SUBSCRIBER_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["email"]) 
        writer.writerow([email])
    
    return True, "Subscribed successfully!"