
import pandas as pd
import numpy as np
from datetime import timedelta

def process_rfq_data(df):
    # Convert RFQTime to datetime if not already
    df['RFQTime'] = pd.to_datetime(df['RFQTime'], errors='coerce')

    # Fill GroupID and Weight columns with default values
    df['GroupID'] = np.nan
    df['Weight'] = np.nan

    # Define group columns
    group_columns = ['CurrencyPair', 'ClientName', 'Product', 'Venue', 'ClientDir', 'RFQTime']

    # Invalid rows where any group column or time is blank/NA/etc.
    invalid_mask = df[group_columns].isnull().any(axis=1) | df[group_columns].isin(['', 'NA', 'None']).any(axis=1)
    df.loc[invalid_mask, 'GroupID'] = -1
    df.loc[invalid_mask, 'Weight'] = 1

    # Process valid rows
    valid_df = df[~invalid_mask].copy()
    group_id_counter = 1
    updated_rows = []

    # Group valid data by super groups
    super_groups = valid_df.groupby(['CurrencyPair', 'ClientName', 'Product', 'Venue', 'ClientDir'])

    for group_key, group_df in super_groups:
        group_df = group_df.sort_values('RFQTime')

        while not group_df.empty:
            base_time = group_df.iloc[0]['RFQTime']
            window_end = base_time + timedelta(minutes=10)
            in_window = group_df[(group_df['RFQTime'] >= base_time) & (group_df['RFQTime'] <= window_end)]

            # Separate booked RFQs if there are more than 1
            booked_rfqs = in_window[in_window['BookingStatus'] == 'Booked']
            if len(booked_rfqs) > 1:
                # Keep only 1 booked in the group
                keep_booked = booked_rfqs.iloc[[0]]
                extra_booked = booked_rfqs.iloc[1:]
                non_booked = in_window[in_window['BookingStatus'] != 'Booked']

                # Group = keep_booked + non_booked
                group_rfq = pd.concat([keep_booked, non_booked])
                n = len(group_rfq)
                weights = [round(1 / n, 4)] * (n - 1)
                weights.append(round(1 - sum(weights), 4))
                group_rfq['GroupID'] = group_id_counter
                group_rfq['Weight'] = weights
                updated_rows.append(group_rfq)
                group_id_counter += 1

                # Remaining booked each get their own group
                for _, row in extra_booked.iterrows():
                    row['GroupID'] = group_id_counter
                    row['Weight'] = 1
                    updated_rows.append(pd.DataFrame([row]))
                    group_id_counter += 1
            else:
                # Regular group
                n = len(in_window)
                weights = [round(1 / n, 4)] * (n - 1)
                weights.append(round(1 - sum(weights), 4))
                in_window['GroupID'] = group_id_counter
                in_window['Weight'] = weights
                updated_rows.append(in_window)
                group_id_counter += 1

            # Remove processed RFQs
            group_df = group_df[~group_df['RFQID'].isin(in_window['RFQID'])]

    # Combine everything back
    final_df = pd.concat([df[invalid_mask]] + updated_rows, ignore_index=True)
    return final_df

# Use the function with dummy file paths
def update_csv(input_path='your_input_file.csv', output_path='your_output_file.csv'):
    df = pd.read_csv(input_path)
    updated_df = process_rfq_data(df)
    updated_df.to_csv(output_path, index=False)
    print(f"✅ File processed and saved to: {output_path}")

# Example usage (replace with your actual file paths)
update_csv('your_input_file.csv', 'your_output_file.csv')