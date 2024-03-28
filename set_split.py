import json
import random

def split_raw_data(test_size:int = 5):
    test_set = {"data":[]}

    with open("./raw_data.json", "r", encoding="utf-8") as r:
        rawdata = json.load(r)

    idx = -1
    for i in range(test_size):
        idx = random.randint(0, len(rawdata["data"])-1)
        test_set["data"].append(rawdata["data"][idx])
        del rawdata["data"][idx]


    with open("train_data.json", "w", encoding="utf-8") as w:
        json.dump(rawdata, w)

    with open("test_data.json", "w", encoding="utf-8") as w:
        json.dump(test_set, w)
        
