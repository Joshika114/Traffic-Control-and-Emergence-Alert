import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# 1) Prepare DataFrame
df = ddf.copy()
assert 'IS_ACCIDENT_ZONE' in df.columns

# Ensure total_vehicles exists
if 'total_vehicles' not in df:
    to_sum = [c for c in ['NUM_UNITS','INJURIES_TOTAL'] if c in df]
    df['total_vehicles'] = df[to_sum].sum(axis=1)

# Select numeric features (excluding the target)
numeric = df.select_dtypes(include='number').columns.drop('IS_ACCIDENT_ZONE')
X = df[numeric]
y = df['IS_ACCIDENT_ZONE']

# 2) Impute & train RandomForest
imp = SimpleImputer(strategy='mean')
X_imp = imp.fit_transform(X)

rf = RandomForestClassifier(n_estimators=30, random_state=42).fit(X_imp, y)
importances = pd.Series(rf.feature_importances_, index=numeric).sort_values(ascending=False)

# 3) Class distribution
dist = y.value_counts(normalize=True)
labels = dist.index.astype(str)
values = dist.values

# 4) Build dashboard (NUM_UNITS vs INJURIES_TOTAL)
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("Feature Importance","NUM_UNITS vs INJURIES_TOTAL","Class Distribution"),
    specs=[[{"type":"bar"},{"type":"scatter"},{"type":"domain"}]]
)

# Panel 1: feature importances
fig.add_trace(
    go.Bar(x=importances.values, y=importances.index, orientation='h',
           marker=dict(color=importances.values, colorscale='Viridis')),
    row=1, col=1
)

# Panel 2: scatter NUM_UNITS vs INJURIES_TOTAL
fig.add_trace(
    go.Scatter(
        x=df['NUM_UNITS'], y=df['INJURIES_TOTAL'],
        mode='markers',
        marker=dict(color=y, colorscale='Turbo', showscale=True, colorbar=dict(title="Zone")),
        text=y
    ), row=1, col=2
)

# Panel 3: donut chart
fig.add_trace(
    go.Pie(labels=labels, values=values, hole=0.5, textinfo='percent+label'),
    row=1, col=3
)

fig.update_layout(
    title="📊 Accident‑Zone Classification Dashboard",
    height=600, width=1200,
    showlegend=False
)

fig.show()
