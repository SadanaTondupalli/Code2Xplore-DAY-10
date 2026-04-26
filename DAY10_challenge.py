import random
import copy
import math
import numpy as np
import pandas as pd

def generate_data(n=15):
    data = []
    for i in range(n):
        data.append({
            "zone": i + 1,
            "metrics": {
                "traffic": random.randint(50, 200),
                "pollution": random.randint(30, 150),
                "energy": random.randint(40, 180)
            },
            "history": [random.randint(20, 100) for _ in range(5)]
        })
    return data

def personalize_data(data, roll):
    if roll % 2 == 0:
        return list(reversed(data))
    else:
        return data[3:] + data[:3]

def custom_risk_score(t, p, e):
    return math.log(t + p + e) + math.sqrt(e)


def mutate_data(data):
    for d in data:
        d["metrics"]["traffic"] += 10
        d["history"].append(random.randint(50, 120))

        t = d["metrics"]["traffic"]
        p = d["metrics"]["pollution"]
        e = d["metrics"]["energy"]

        d["risk"] = custom_risk_score(t, p, e)

def to_dataframe(data):
    rows = []
    for d in data:
        rows.append({
            "zone": d["zone"],
            "traffic": d["metrics"]["traffic"],
            "pollution": d["metrics"]["pollution"],
            "energy": d["metrics"]["energy"],
            "risk": d.get("risk", 0)
        })
    return pd.DataFrame(rows)


def manual_corr(x, y):
    mx, my = np.mean(x), np.mean(y)
    num = np.sum((x - mx) * (y - my))
    den = math.sqrt(np.sum((x - mx)**2) * np.sum((y - my)**2))
    return num / den

def analyze(df):
    mean = df["risk"].mean()
    std = df["risk"].std()

    anomalies = df[df["risk"] > mean + std]

    variance = df["risk"].var()
    stability = 1 / variance if variance != 0 else 0

    corr = manual_corr(df["traffic"].values, df["energy"].values)

    return anomalies, stability, corr

def detect_clusters(df):
    risky = df["risk"] > df["risk"].mean()
    clusters, temp = [], []

    for i in range(len(risky)):
        if risky[i]:
            temp.append(int(df.iloc[i]["zone"]))
        else:
            if len(temp) > 1:
                clusters.append(temp)
            temp = []
    return clusters


roll_number = 24110011738

data = generate_data()
data = personalize_data(data, roll_number)

assign_copy = data
shallow_copy = copy.copy(data)
deep_copy = copy.deepcopy(data)

print("\nBEFORE MUTATION")
print("ORIGINAL:", data[0])
print("SHALLOW :", shallow_copy[0])
print("DEEP    :", deep_copy[0])

mutate_data(shallow_copy)

print("\nAFTER MUTATION (Shallow Copy)")
print("ORIGINAL:", data[0])       # CORRUPTED
print("SHALLOW :", shallow_copy[0])
print("DEEP    :", deep_copy[0]) # SAFE

df = to_dataframe(data)

print("\nDATAFRAME")
print(df)

anomalies, stability_index, corr = analyze(df)

print("\nANOMALY ZONES")
print(anomalies)

print("\nMANUAL CORRELATION (traffic vs energy)")
print(corr)
clusters = detect_clusters(df)

print("\nRISK CLUSTERS")
print(clusters)

max_risk = df["risk"].max()
min_risk = df["risk"].min()

result = (max_risk, min_risk, stability_index)

print("\nRESULT TUPLE")
print(result)

if stability_index > 1:
    decision = "System Stable"
elif stability_index > 0.5:
    decision = "Moderate Risk"
elif stability_index > 0.2:
    decision = "High Corruption Risk"
else:
    decision = "Critical Failure"

print("\nFINAL DECISION")
print(decision)