import random
import re
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

def refactor_response(response):
    if (re.search(pattern, response)):
        res = re.findall(pattern, response)
        for var in res:
            #a smart/jugaad to turn a string to a variable name I is proud of this :)
            want = globals()[(var.replace('{', '').replace('}', ''))]
            response = response.replace(var, want)
    return response

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "SmartBoi"
print(f"Start chatting with {bot_name}! (type 'quit' to exit)")
falseCounter = 0
name = input("Please enter your name: ")
call = '8073982643'
link = 'reflecton.in'
location = 'g.maps'
locationlink = 'g.maps'
appointmentlink = 'appoint.link'
pattern = r"{[a-z]+\}"
want = ''
while falseCounter<3:
    # sentence = "do you use credit cards?"
    sentence = input("You: ")
    if sentence == "quit":
        for intent in intents['intents']:
            if 'goodbye' == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
                
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == 'goodbye' and tag == intent["tag"]:
                response = str(random.choice(intent['responses']))
                rresponse = refactor_response(response)
                print(f"{bot_name}: {rresponse}")
                print("\nType a question to continue")
                choice = input("Press enter to confirm you want to quit: ")
                if not choice:
                    exit()
            if tag == intent["tag"]:
                response = str(random.choice(intent['responses']))
                rresponse = refactor_response(response)
                print(f"{bot_name}: {rresponse}")
    else:
        print(f"{bot_name}: I am {prob * 100:.2f}% sure that you're talking about '{tag}'. Need more training...")
        falseCounter += 1
if falseCounter>=3:
    print("Auto quitting. AI needs more training...")
