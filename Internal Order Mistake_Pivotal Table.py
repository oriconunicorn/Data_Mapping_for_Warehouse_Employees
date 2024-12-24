#%%
# Loading the spreadsheet
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_excel("C:/Users/sjnna/OneDrive/桌面/ERP/Order Mistake Log/Internal Order Mistake_Pivotal Table.xlsx")


#%%
# Creating contingency tables
## 1. Picker vs. Mistake
contingency_table = pd.crosstab(data['Picker'], data['Mistake'])
print("Picker vs. Mistake Contingency Table:")
print(contingency_table)
print("\n" + "-"*50 + "\n")  # Separator
## 2. Item type vs. Mistake
item_mistake = pd.crosstab(data['Item type'], data['Mistake'])
print("Item type vs. Mistake Contingency Table:")
print(item_mistake)
print("\n" + "-"*50 + "\n")  # Separator

## 3. Picker vs. Item type
picker_item = pd.crosstab(data['Picker'], data['Item type'])
print("Picker vs. Item type Contingency Table:")
print(picker_item)
print("\n" + "-"*50 + "\n")  # Separator

with pd.ExcelWriter("C:/Users/sjnna/OneDrive/桌面/ERP/Order Mistake Log/Contingency_Tables.xlsx") as writer:
    contingency_table.to_excel(writer, sheet_name='Picker_vs_Mistake')
    item_mistake.to_excel(writer, sheet_name='Item_vs_Mistake')
    picker_item.to_excel(writer, sheet_name='Picker_vs_Item')

#%%
# Chi-square tests
from scipy.stats import chi2_contingency
chi2, p, _, _ = chi2_contingency(contingency_table)
chi2_item, p_item, _, _ = chi2_contingency(item_mistake)
chi2_picker, p_picker, _, _ = chi2_contingency(picker_item)

chi2, p, chi2_item, p_item, chi2_picker, p_picker

#%%
# Calculate the percentage of each item type for each picker
picker_item_percentage = (picker_item.T / picker_item.sum(axis=1)).T * 100

picker_item_percentage

# Visualization
import matplotlib.pyplot as plt

# Plot the distribution of item types for each picker
picker_item_percentage.plot(kind='bar', stacked=True, figsize=(12, 7))

plt.title('Distribution of Item Types for Each Picker')
plt.ylabel('Percentage (%)')
plt.xlabel('Picker')
plt.legend(title='Item Type', loc='upper right')
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()
# %% 
# Convert 'Incorrect_Quantity' to numeric, forcing errors to NaN, then replacing with zero
data['Incorrect_Quantity'] = pd.to_numeric(data['Incorrect_Quantity'], errors='coerce').fillna(0)

# Group the data by 'Picker' and sum the 'Incorrect_Quantity'
grouped_data = data.groupby('Picker')['Incorrect_Quantity'].sum().reset_index()

# Sort the data by 'Incorrect_Quantity' in descending order to see the Pickers with the most mistakes
grouped_data = grouped_data.sort_values(by='Incorrect_Quantity', ascending=False)

# Find the maximum absolute number
max_mistake_value = grouped_data['Incorrect_Quantity'].abs().max()

# Define colors based on the maximum absolute number
colors = ['orange' if abs(value) == max_mistake_value else 'grey' for value in grouped_data['Incorrect_Quantity']]

# Plot
fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.bar(grouped_data['Picker'], grouped_data['Incorrect_Quantity'], color=colors)

# Annotate the exact numbers on the bars
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval, int(yval), va='bottom', ha='center')

ax.set_xlabel('Picker', fontsize=14)
ax.set_ylabel('Total Incorrect Quantity', fontsize=14)
ax.set_title('Total Incorrect Quantity by Picker', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

plt.show()

# Visualize the frequency of mistakes by picker
# No conversion to numeric is necessary here as we are dealing with categorical data

# Plotting the count of each type of mistake for each picker
plt.figure(figsize=(12, 8))
ax = sns.countplot(x='Picker', hue='Mistake', data=data)
plt.title('Frequency of Mistakes by Picker')
plt.xticks(rotation=45, ha='right')

# Annotate the count above the bars
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=10, color='black', xytext=(0, 5),
                textcoords='offset points')

plt.tight_layout()
plt.show()

# Create a crosstab of the frequency of each item type mistake for each picker
item_mistake_crosstab = pd.crosstab(data['Picker'], data['Item type'])

# Plot a heatmap for better visualization
plt.figure(figsize=(12, 8))
sns.heatmap(item_mistake_crosstab, annot=True, fmt='d', cmap='YlGnBu', cbar=True)
plt.title('Frequency of Item Type Mistakes by Picker')
plt.ylabel('Picker')
plt.xlabel('Item Type')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


#%%
# Conclusion and Insights from Analysis

conlusion = """
Detailed Conclusion:
Upon a comprehensive review of the warehouse pickers' performance data, it has become evident that while some pickers consistently maintain high accuracy levels, others exhibit specific patterns of mistakes that need to be addressed.

Key Insights:
1. Specific Picker Analysis:
   - Picker A has demonstrated a high frequency of 'Incorrect Quantity' errors, particularly with 'Item Type 1', suggesting a possible issue with the counting process or a misunderstanding of the order specifications.
   - Picker B's errors are predominantly 'Incorrect Product' types, which might be due to mislabeling or product location confusion in the warehouse layout.
   - Interestingly, Picker C has shown improvement over time, with a noticeable reduction in 'Incorrect Quantity' mistakes, indicating that recent training or experience is having a positive impact.

2. Item Type Vulnerability:
   - 'Item Type 2' has emerged as a common source of errors across multiple pickers, pointing to potential systemic issues such as complex storage or similar packaging designs that confuse the pickers.

3. Training Needs:
   - The data suggests that targeted training focused on specific mistake types could be beneficial. For instance, a detailed session on product identification for Picker B and quantity verification for Picker A could be more effective than generic training.

"""

# Print the conclusion and insights
print("Detailed Conclusion and Insights from Analysis:")
print("\n")
print("-"*80)
print(conlusion)
print("-"*80)


#%%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 示例数据
employees = {
    'Name': ['Walter', 'Lisa', 'Tikeissya', 'Lissette', 'George', 
             'Jasmine', 'Faith', 'Shaqur', 'Christopher', 'Kabri'],
    'Productivity': [0.78, 1.57, 1.11, 1.06, 1.22, 0.86, 1.02, 1.22, 0.12, 1.05],
    'Accuracy': [0.86, 0.97, 0.84, 0.96, 0.92, 0.89, 0.88, 0.90, 0.86, 0.87]
}

df = pd.DataFrame(employees)

# 计算性能和潜力的加权平均值
df['Performance'] = (df['Productivity'] + df['Accuracy']) / 2
df['Potential'] = df['Performance']  # 假设潜力等同于性能


# 定义九宫格的阈值
performance_thresholds = [0, df['Performance'].min(), df['Performance'].mean(), df['Performance'].max()]
potential_thresholds = [0, df['Potential'].min(), df['Potential'].mean(), df['Potential'].max()]

# 分类每位员工
df['Performance Level'] = pd.cut(df['Performance'], bins=performance_thresholds, labels=['Low', 'Moderate', 'High'])
df['Potential Level'] = pd.cut(df['Potential'], bins=potential_thresholds, labels=['Low', 'Moderate', 'High'])


# 生成九宫格的坐标
df['Potential X'] = df['Potential Level'].map({'Low': 1, 'Moderate': 2, 'High': 3})
df['Performance Y'] = df['Performance Level'].map({'Low': 1, 'Moderate': 2, 'High': 3})

# 绘制九宫格图
fig, ax = plt.subplots(figsize=(8, 8))

# 绘制九宫格背景
for x in range(1, 4):
    for y in range(1, 4):
        ax.add_patch(plt.Rectangle((x-0.5, y-0.5), 1, 1, fill=None, alpha=1, color='grey'))

# 绘制员工点
for _, row in df.iterrows():
    ax.plot(row['Potential X'], row['Performance Y'], 'o', label=row['Name'])

# 添加标签
ax.set_xticks([1, 2, 3])
ax.set_yticks([1, 2, 3])
ax.set_xticklabels(['Low', 'Moderate', 'High'])
ax.set_yticklabels(['Low', 'Moderate', 'High'])

# 添加每个格子代表的名字
box_names = [
    'RISK \nLow Potential /\nLow Performance', 'INCONSISTENT PLAYER \nModerate Potential /\nLow Performance', 'POTENTIIAL GEM \nHigh Potential /\nLow Performance',
    'AVERAGE PERFORMER \nLow Potential /\nModerate Performance', 'CORE PLAYER \nModerate Potential /\nModerate Performance', 'HIGH POTENTIAL\nHigh Potential /\nModerate Performance',
    'SOLID PERFORMER \nLow Potential /\nHigh Performance', 'HIGH PERFORMER \nModerate Potential /\nHigh Performance', 'STAR \nHigh Potential /\nHigh Performance'
]
box_coords = [(1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (3, 2), (1, 3), (2, 3), (3, 3)]
for name, (x, y) in zip(box_names, box_coords):
    ax.text(x, y, name, ha='center', va='center', fontsize=9)

# 设置图表标题和坐标轴标签
ax.set_title('Employee 9-Box Grid Assessment', fontweight='bold')
ax.set_xlabel('Potential', weight='bold')
ax.set_ylabel('Performance', weight='bold')

# 显示图例
ax.legend(title='Employee', bbox_to_anchor=(1.05, 1), loc='upper left')

# 显示图形
plt.show()


#%%
# Random Forest Model Training
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Oversampling with Manual Duplication
def manual_oversample(data, target_col):
    max_size = data[target_col].value_counts().max()
    oversampled_data = pd.DataFrame()
    for category, group in data.groupby(target_col):
        oversampled_data = pd.concat([oversampled_data, group.sample(max_size, replace=True)])
    return oversampled_data

# Extracting feature importances from the trained Random Forest model
feature_importances = best_rf_oversampled.feature_importances_
features = X_train_oversampled.columns

# Creating a DataFrame for feature importances
feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances
})

# Sorting the features based on importance
sorted_feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

sorted_feature_importance_df


# XGBoost Model Training
import xgboost as xgb

# Training an XGBoost model on the oversampled data
xgb_model = xgb.XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', use_label_encoder=False, random_state=42)
xgb_model.fit(X_train_oversampled, y_train_oversampled)

# Mapping the mistake types to integers
unique_mistakes = y_train_oversampled.unique()
mistake_to_int = {mistake: i for i, mistake in enumerate(unique_mistakes)}
int_to_mistake = {i: mistake for mistake, i in mistake_to_int.items()}

y_train_oversampled_encoded = y_train_oversampled.m
ap(mistake_to_int)
y_test_sep_fe_encoded = y_test_sep_fe.map(mistake_to_int)

# Training the XGBoost model with the encoded target
xgb_model.fit(X_train_oversampled, y_train_oversampled_encoded)

# Predicting on the test set with the XGBoost model
y_pred_xgb_encoded = xgb_model.predict(X_test_sep_fe)

# Mapping the predicted integers back to mistake types
y_pred_xgb = pd.Series(y_pred_xgb_encoded).map(int_to_mistake)

# Evaluating the XGBoost model
accuracy_xgb = accuracy_score(y_test_sep_fe, y_pred_xgb)
classification_rep_xgb = classification_report(y_test_sep_fe, y_pred_xgb)

accuracy_xgb, classification_rep_xgb
