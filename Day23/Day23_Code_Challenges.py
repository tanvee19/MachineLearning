"""Code Challenge:
dataset: BreadBasket_DMS.csv

Q1. In this code challenge, you are given a dataset which has data and time
 wise transaction on a bakery retail store.
1. Draw the pie chart of top 15 selling items.
2. Find the associations of items where min support should be 0.0025, 
min_confidence=0.2, min_lift=3.
3. Out of given results sets, show only names of the associated item
 from given result row wise.
"""

import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori

dataset = pd.read_csv('BreadBasket_DMS.csv')

d = dataset["Item"].value_counts().head(15)
plt.pie(d.values,explode = None,labels = d.index,colors = ['red','green','aqua','red','blue','purple','orange','red','green','blue','purple','orange','black','white','green'] )


dataset = dataset.mask(dataset.eq("NONE")).dropna()

def sort(values):
    s = ','.join(values)
    return s    
df = dataset.groupby("Transaction")["Item"].apply(sort)
"""
transactions = []
for j in range(len(df)):
    transactions.append(list(df.values[j].split(',')))
    
"""    
rules = list(apriori(df, min_support = 0.0025, min_confidence = 0.2, min_lift = 3))



for item in rules:

    # first index of the inner list
    # Contains base item and add item
    pair = item[0] 
    items = [x for x in pair]
    print("Rule: " + items[0] + " -> " + items[1])

    #second index of the inner list
    print("Support: " + str(item[1]))

    #third index of the list located at 0th
    #of the third index of the inner list

    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")
    
    
"""
Code Challenge:
Datset: Market_Basket_Optimization.csv
Q2. In today's demo sesssion, we did not handle the null values before
 fitting the data to model, remove the null values from each row and 
 perform the associations once again.
Also draw the bar chart of top 10 edibles.
"""


import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori

# Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)


transactions = []
l = []
dataset = dataset.fillna("None")
for i in range(len(dataset)+1):
    transactions.append([])

for j in range(len(list(dataset.columns))):
    for i in range(0, len(dataset)):
        if dataset[j][i] != "None":
            transactions[i].append(dataset[j][i])
            l.append(dataset[j][i])
        else:
            pass
    
    
    
# Training Apriori on the dataset

rules = apriori(transactions, min_support = 0.003, min_confidence = 0.25, min_lift = 4)

# Visualising the results
results = list(rules)




for item in results:

    # first index of the inner list
    # Contains base item and add item
    pair = item[0] 
    items = [x for x in pair]
    print("Rule: " + items[0] + " -> " + items[1])

    #second index of the inner list
    print("Support: " + str(item[1]))

    #third index of the list located at 0th
    #of the third index of the inner list

    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")

d   = pd.DataFrame(l,)

d1 = d[0].value_counts().head(10)

plt.bar(d,d1 )
