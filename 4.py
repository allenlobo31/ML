import pandas as pd


dataset = {
    'Sky': ['Suny', 'Suny', 'Rainy', 'Suny'],
    'AirTemp': ['Warm', 'Cold', 'Warm', 'Warm'],
    'Humidity': ['High', 'Normal', 'High', 'High'],
    'Wind': ['Strong', 'Strong', 'Medium', 'Strong'],
    'Water': ['Warm', 'Warm', 'Warm', 'Cool'],
    'Output': ['Yes', 'Yes', 'No', 'Yes']
}

df = pd.DataFrame(dataset)


def find_s_algorithm(df):
    """Implementing it"""

    attribute = df.iloc[:, :-1].values
    data = df.iloc[:, -1].values

    for i in range(len(data)):
        if data[i] == "Yes":
            hypothesis = attribute[i].copy()
            break

    for i in range(len(data)):
        if data[i] == "Yes":
            for j in range(len(hypothesis)):
                if hypothesis[j] != attribute[i][j]:
                    hypothesis[j] = "?"

    return hypothesis


final_hypothesis = find_s_algorithm(df)

print("Most Specific Hypothesis:", final_hypothesis)

