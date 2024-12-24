#%%
import pandas as pd

# Load the provided spreadsheet
shipment_stats_df = pd.read_excel("C:/Users/sjnna/OneDrive/桌面/ERP/Order Mistake Log/All Group members shipment statistics(Daily sales orders).xlsx")

# Adjusting the column names based on the values in the first row
shipment_stats_df.columns = shipment_stats_df.iloc[0]
shipment_stats_df = shipment_stats_df.drop(0).reset_index(drop=True)

#%%
# Define the mapping dictionaries
pickers = ["Lisa",
           "Lissette", 
           "George", 
           "Faith",
           "Nakita",
           "Ebonie",
           "Rodney",
           "Rossie",
           "Donaja",
           "Tamiki",
           "Reginald",
           "Kabine",
           "Delviliz",
           "Felix",
           "Korey"]
picker_accounts = ["Lisa.Xie",
                   "Lissette.Colon", 
                   "George.Groover", 
                   "Faith.Timmons", 
                   "Nicketa.Burnett",
                   "Ebonie.Gibbs",
                   "Rodney.Bonner",
                   "Rossie.Ashlock",
                   "Donaja.Lum",
                   "Tamiki.Weldon",
                   "Reginald.Victor",
                   "Kabine.Toure",
                   "Delviliz.Rivera",
                   "Felix.Diaz",
                   "Korey.Martin"]
picker_mapping = dict(zip(picker_accounts, pickers))

checkers = ["Jenitza", 
            "Hao", 
            "Jeffrey",  
            "Sydney",
            "Keishla",
            "Sarah",
            "John",
            "Shameeka",
            "Ski",
            "Alexandria",
            "Britney",
            "Mark",
            "Chrissy"]

checker_accounts = ["Jenitza.Medina.Gomez", 
                    "Vinh.Hao.Luu",  
                    "Jeffrey.Caraballo", 
                    "Vannessa.Colon", 
                    "Keishla.Gonzalez",
                    "Sarah.Schaech",
                    "John.Rodriguez",
                    "Shameeka.Jackson",
                    "Ski.Nolan",
                    "Alexandria.Bridgett",
                    "Britney.Gass",
                    "Mark.Johnson",
                    "Chrissy.Ortiz"]
checker_mapping = dict(zip(checker_accounts, checkers))

# 特殊checker的账号名称
special_checkers_accounts = ["Vinh.Hao.Luu", "Jeffrey.Caraballo", "Alexandria.Bridgett"]

# Filter the dataframe for pickers
picker_filtered_df = shipment_stats_df[shipment_stats_df['Account Name'].isin(picker_accounts)]
picker_filtered_df['Picker'] = picker_filtered_df['Account Name'].map(picker_mapping)
picker_filtered_df = picker_filtered_df[['Picker', 'checked Order Qty', 'checked products qty']]

# Filter the dataframe for checkers
checker_filtered_df = shipment_stats_df[shipment_stats_df['Account Name'].isin(checker_accounts)]
checker_filtered_df['Checker'] = checker_filtered_df['Account Name'].map(checker_mapping)

# 特殊checker的姓名
special_checkers = [checker_mapping[account] for account in special_checkers_accounts if account in checker_mapping]

# 创建一个条件列，基于checker是否为特殊checker来决定使用哪个列的数据
checker_filtered_df['Rechecked Qty'] = checker_filtered_df.apply(
    lambda x: x['shipping products qty'] if x['Checker'] in special_checkers else x['Rechecked products qty'],
    axis=1
)

checker_filtered_df = checker_filtered_df[['Checker', 'Rechecked order number (order)', 'Rechecked Qty']]

# Define the saving paths
picker_save_path = "C:/Users/sjnna/OneDrive/桌面/ERP/Order Mistake Log/Picker_Data.xlsx"
checker_save_path = "C:/Users/sjnna/OneDrive/桌面/ERP/Order Mistake Log/Checker_Data.xlsx"

# Save the dataframes to Excel files
picker_filtered_df.to_excel(picker_save_path, index=False)
checker_filtered_df.to_excel(checker_save_path, index=False)

