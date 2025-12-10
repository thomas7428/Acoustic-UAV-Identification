from termcolor import colored
import numpy as np
import json
import sys
from pathlib import Path

# Import centralized configuration
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Soft Voting Section

# Output results (storing voting model performance results).
RESULTS = config.VOTED_RESULTS_PATH_STR
VOTED_MODELS_SCORES = "Hard_Voting_CNN/run_10/FINAL_SCORES_real_world.json"  # Hard voting performance scores output.

# json files with model confidence results (10 models).
cert_1 = config.get_path_str(config.VOTED_PATHS[1])
cert_2 = config.get_path_str(config.VOTED_PATHS[2])
cert_3 = config.get_path_str(config.VOTED_PATHS[3])
cert_4 = config.get_path_str(config.VOTED_PATHS[4])
cert_5 = config.get_path_str(config.VOTED_PATHS[5])
cert_6 = config.get_path_str(config.VOTED_PATHS[6])
cert_7 = config.get_path_str(config.VOTED_PATHS[7])
cert_8 = config.get_path_str(config.VOTED_PATHS[8])
cert_9 = config.get_path_str(config.VOTED_PATHS[9])
cert_10 = config.get_path_str(config.VOTED_PATHS[10])


def vote_result(json_path):

    # Dictionary to store data.
    results = {
        "combined_votes": [],
        "voted_predictions": [],
    }

    with open(cert_1) as a, open(cert_2) as b, open(cert_3) as c, open(cert_4) as d, open(cert_5) as e, open(cert_6) as f, open(cert_7) as g, open(cert_8) as h, open(cert_9) as j, open(cert_10) as k:
        data_1 = json.load(a)
        data_2 = json.load(b)
        data_3 = json.load(c)
        data_4 = json.load(d)
        data_5 = json.load(e)
        data_6 = json.load(f)
        data_7 = json.load(g)
        data_8 = json.load(h)
        data_9 = json.load(j)
        data_10 = json.load(k)

    l = np.array(data_1["results"])
    m = np.array(data_2["results"])
    n = np.array(data_3["results"])
    o = np.array(data_4["results"])
    p = np.array(data_5["results"])
    q = np.array(data_6["results"])
    r = np.array(data_7["results"])
    s = np.array(data_8["results"])
    t = np.array(data_9["results"])
    u = np.array(data_10["results"])

    # Summing predictions.
    z = l + m + n + o + p + q + r + s + t + u

    print(z)

    # Finalising prediction output from voted models.
    for i in z:
        i = round(i, 10)
        if i > 10 or i < 0:
            print(i)
            print(colored("Error: The average value has been computed incorrectly. Please check.", "red"))
            sys.exit(1)
        elif i >= 5:
            vote = 1
        else:
            vote = 0
        results["combined_votes"].append(int(i))
        results["voted_predictions"].append(int(vote))

    with open(json_path, "w") as fp:
        json.dump(results, fp, indent=4)

def performance_calcs(performance_path):
    # Dictionary to store data.
    performance = {
        "TP": [],
        "FN": [],
        "TN": [],
        "FP": [],
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "F1 Score": [],
    }

    with open(RESULTS, "r") as fp:
        data = json.load(fp)

    # Convert lists to numpy arrays  --> same process as single model performance predicts.
    y = np.array(data["voted_predictions"])

    a = float(sum(y[0:int(len(y)/2)]))
    b = float(sum(y[int(len(y)/2):int(len(y))]))

    TP = a
    FN = int(len(y)/2) - a
    FP = b
    TN = int(len(y)/2) - b

    Accuracy = (TP + TN) / (TP + TN + FN + FP)
    Precision = TP/(TP+FP)
    Recall = TP/(TP+FN)
    F1 = (2*Precision*Recall)/(Precision+Recall)

    performance["TP"].append(TP)
    performance["FN"].append(FN)
    performance["TN"].append(TN)
    performance["FP"].append(FP)
    performance["Accuracy"].append(Accuracy)
    performance["Precision"].append(Precision)
    performance["Recall"].append(Recall)
    performance["F1 Score"].append(F1)


    with open(performance_path, "w") as fp:
        json.dump(performance, fp, indent=4)

if __name__ == "__main__":
    vote_result(RESULTS)
    performance_calcs(VOTED_MODELS_SCORES)

    print(colored("Process completed successfully!", "green"))