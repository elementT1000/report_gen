import plotly.graph_objects as go
import pandas as pd

# Create a sample DataFrame
df = pd.DataFrame({'col1': [1, 2, 3, 4, 5], 'col2': [6, 7, 8, 9, 10], 'col3': [11, 12, 13, 14, 15]})

# Calculate the mean and standard deviation
df['mean'] = df.mean(axis=1)
df['std'] = df.std(axis=1)

# Create the figure
fig = go.Figure()

# Add the primary line indicating the mean
fig.add_trace(go.Scatter(x=df.index, y=df['mean'], mode='lines', name='Mean', line=dict(color='blue', width=2)))

# Add the faint lines indicating the standard deviation
fig.add_trace(go.Scatter(x=df.index, y=df['mean']+df['std'], mode='lines', name='+1 Std Dev', line=dict(color='gray', width=1, dash='dot')))
fig.add_trace(go.Scatter(x=df.index, y=df['mean']-df['std'], mode='lines', name='-1 Std Dev', line=dict(color='gray', width=1, dash='dot')))

# Show the figure
fig.show()