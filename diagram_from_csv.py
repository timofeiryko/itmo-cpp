import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
data = pd.read_csv('is_cpp.csv')

# Check if the data is loaded correctly
print(data)


plt.figure(figsize=(10, 6))
plt.bar(data['is_cpp'], data['count'], color='skyblue')

# Add titles and labels
plt.title('Bar Diagram from CSV Data')
plt.xlabel('is cpp')
plt.ylabel('count')

# Show the plot
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('is_cpp.png') 
plt.show()
