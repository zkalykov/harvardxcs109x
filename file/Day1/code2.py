import pandas as pd
import matplotlib.pyplot as plt

data = {
    'TV': [230.1, 44.5, 17.2, 151.5, 180.8],
    'Radio': [37.8, 39.3, 45.9, 41.3, 10.8],
    'Newspaper': [69.2, 45.1, 69.3, 58.5, 58.4],
    'Sales': [22.1, 10.4, 9.3, 18.5, 12.9]
}

df = pd.DataFrame(data)

fig, ax = plt.subplots()

ax.scatter(df[['TV', 'Radio', 'Newspaper']].sum(axis=1), df['Sales'])
ax.set_title('Total Advertising Spend vs Sales')
ax.set_xlabel('Total Advertising Spend ($1000s)')
ax.set_ylabel('Sales ($1000s)')

plt.show()
