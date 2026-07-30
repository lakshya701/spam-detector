"""
Builds data/sms_dataset.csv — a labeled spam/ham SMS dataset.

Note: this sandbox has no internet access, so the real SMS Spam Collection
(UCI/Kaggle) can't be downloaded here. This script generates a
template-based dataset that mirrors the same real-world spam/ham patterns
(free offers, urgency, prize claims, phone-number call-to-action for spam;
plain scheduling/small-talk for ham) so the full pipeline runs end-to-end.

To use the real dataset instead: download SMSSpamCollection from
https://archive.ics.uci.edu/dataset/228/sms+spam+collection, save it as
data/sms_dataset.csv with columns `label,message`, and skip this script.
"""

import random
import csv
import os

random.seed(42)

spam_templates = [
    "Congratulations! You've WON a {prize}! Claim now by calling {phone} before it expires.",
    "URGENT: Your account will be suspended. Verify now at {link} to avoid charges.",
    "FREE entry to win a {prize}! Text WIN to {shortcode} now, limited time offer!",
    "You have been selected for a {amount} cash reward! Call {phone} to claim TODAY.",
    "Get a FREE {prize} just for signing up! Click {link} now, offer ends tonight.",
    "LAST CHANCE: Claim your {amount} voucher before midnight. Reply YES to {shortcode}.",
    "Your loan of {amount} has been approved! No credit check. Call {phone} now.",
    "WINNER!! As a valued customer you have been selected to receive a {prize}. Call {phone}.",
    "Hot singles in your area want to chat NOW! Click {link} to meet them tonight.",
    "You've been chosen for a FREE {prize} trial. No cost, no obligation. Visit {link}.",
    "ALERT: Unusual activity on your account. Confirm your details at {link} immediately.",
    "Congrats! Your number won {amount} in our weekly draw. Text CLAIM to {shortcode}.",
    "Limited offer: Buy 1 get 1 FREE on {prize}. Shop now at {link}, today only!",
    "Your subscription reward: a FREE {prize} is waiting. Call {phone} within 24 hrs.",
    "Reminder: claim your unclaimed {amount} refund now at {link} before it's forfeited.",
]

ham_templates = [
    "Hey, are we still on for {activity} tomorrow at {time}?",
    "Can you pick up some {item} on your way home?",
    "Meeting got moved to {time}, see you in the conference room.",
    "Thanks for the help with the {item} yesterday, really appreciated it.",
    "Running a bit late, will reach by {time}, sorry about that.",
    "Let's grab {activity} this weekend if you're free.",
    "Don't forget mom's birthday is on {day}, we should plan something.",
    "I sent you the notes for {item}, let me know if it makes sense.",
    "Can we reschedule our call to {time} instead of the morning?",
    "Just landed, will call you once I get to the hotel.",
    "The {item} report is done, sending it over in a bit.",
    "Happy {day}! Hope you have a great one.",
    "Did you finish the assignment on {item}? I'm stuck on the last part.",
    "Let's catch up over {activity} sometime next week.",
    "Reminder: rent is due on {day}, just wanted to flag it.",
]

fillers = {
    "prize": ["iPhone 15", "$500 gift card", "holiday package", "laptop", "smartwatch", "cruise ticket"],
    "phone": ["09012345678", "07098765432", "08123456789", "09876543210"],
    "link": ["bit.ly/claim-now", "secure-verify.win", "freegift-offer.com", "account-update.net"],
    "amount": ["$1000", "$500", "$250", "$5000"],
    "shortcode": ["80888", "60060", "70070", "90909"],
    "activity": ["lunch", "coffee", "dinner", "a movie", "a walk"],
    "time": ["3pm", "6:30pm", "10am", "noon", "8pm"],
    "item": ["groceries", "the project", "milk", "the presentation", "the assignment"],
    "day": ["Friday", "Monday", "Saturday", "the 5th"],
}


def fill(template):
    out = template
    for key, options in fillers.items():
        if "{" + key + "}" in out:
            out = out.replace("{" + key + "}", random.choice(options))
    return out


def generate(n_per_class=100):
    rows = []
    for _ in range(n_per_class):
        t = random.choice(spam_templates)
        rows.append(("spam", fill(t)))
    for _ in range(n_per_class):
        t = random.choice(ham_templates)
        rows.append(("ham", fill(t)))
    random.shuffle(rows)
    return rows


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    rows = generate(n_per_class=120)
    with open("data/sms_dataset.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "message"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to data/sms_dataset.csv")
