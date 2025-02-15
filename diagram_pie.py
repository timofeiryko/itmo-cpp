import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
data = pd.read_csv('mechanism.csv')

# Check if the data is loaded correctly
print(data)


plt.figure(figsize=(10, 6))
labels = data['mechanism']
sizes = data['count']
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.axis('equal')
plt.savefig('mechanism.png')
plt.show()
