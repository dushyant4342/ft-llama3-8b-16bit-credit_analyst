# credit_feature_engineer.py

import pandas as pd
import numpy as np

class CreditFeatureEngineer:
    """
    A class to perform complex feature engineering on raw credit bureau data.
    It takes a DataFrame with duplicate customer_no entries (representing different accounts)
    and generates a rich set of features for analysis.
    """

    def _fill_nulls(self, col):
        """A helper function to intelligently fill null values based on column type."""
        if pd.api.types.is_categorical_dtype(col):
            if 'NA' not in col.cat.categories:
                col = col.cat.add_categories(['NA'])
            return col.fillna('NA')
        elif pd.api.types.is_object_dtype(col):
            return col.fillna('NA')
        elif pd.api.types.is_numeric_dtype(col):
            return col.fillna(0)
        else:
            return col.fillna('Unknown')

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes the raw DataFrame to create DPD, utilization, and other credit-based features.
        """
        final_result = df.copy()

        # --- 1. DPD (Days Past Due) Features ---
        data_cols_x = [f'pay_hist_{i}_x' for i in range(1, 37)]
        data_cols_y = [f'pay_hist_{i}_y' for i in range(1, 37)]

        for months in [36, 24, 18, 12, 6, 3, 2, 1]:
            col_name = f"max_dpd_l{months}m_x" if months > 1 else "max_dpd_cm_x"
            if all(c in final_result.columns for c in data_cols_x[:months]):
                final_result[col_name] = final_result[data_cols_x[:months]].max(axis=1)
                final_result[col_name] = final_result[col_name].fillna(0)

        for months in [36, 24, 18, 12, 6, 3, 2, 1]:
            col_name = f"max_dpd_l{months}m_y" if months > 1 else "max_dpd_cm_y"
            if all(c in final_result.columns for c in data_cols_y[:months]):
                final_result[col_name] = final_result[data_cols_y[:months]].max(axis=1)
                final_result[col_name] = final_result[col_name].fillna(0)
        
        for months in [36, 24, 18, 12, 6, 3, 2, 1]:
            prefix = f"max_dpd_l{months}m" if months > 1 else "max_dpd_cm"
            if f'{prefix}_y' in final_result.columns and f'{prefix}_x' in final_result.columns:
                final_result[f'{prefix}_diff'] = final_result[f'{prefix}_y'] - final_result[f'{prefix}_x']

        # --- 2. DPD Status & Delinquency Features ---
        final_result['string_length_x'] = final_result['pay_status_history_x'].str.len().fillna(0)
        final_result['string_length_y'] = final_result['pay_status_history_y'].str.len().fillna(0)

        final_result['latest_payment_dpd_status_y_adjusted'] = np.where(final_result['string_length_y'] - final_result['string_length_x'] > 1, final_result['latest_payment_dpd_status2_y'], final_result['latest_payment_dpd_status_y'])
        final_result['latest_payment_dpd_status2_y_adjusted'] = np.where(final_result['string_length_y'] - final_result['string_length_x'] > 1, final_result['latest_payment_dpd_status3_y'], final_result['latest_payment_dpd_status2_y'])
        
        final_result['max_delinquency_latest_2_months_y'] = final_result[['latest_payment_dpd_status_y_adjusted', 'latest_payment_dpd_status2_y_adjusted', 'latest_payment_dpd_status_y']].max(axis=1)
        final_result['latest_payment_dpd_status_diff'] = np.where(final_result['string_length_y'] - final_result['string_length_x'] > 1, final_result['max_delinquency_latest_2_months_y'] - final_result['latest_payment_dpd_status_x'], final_result['latest_payment_dpd_status_y'] - final_result['latest_payment_dpd_status_x'])

        data_cols_x_max_dpd = ['max_dpd_l36m_diff','max_dpd_l24m_diff','max_dpd_l18m_diff','max_dpd_l12m_diff','max_dpd_l6m_diff','max_dpd_l3m_diff','max_dpd_cm_diff']
        final_result['max_delinquency_detected'] = final_result[data_cols_x_max_dpd].max(axis=1)
        final_result['min_delinquency_detected'] = final_result[data_cols_x_max_dpd].min(axis=1)

        # --- 3. Utilization Features (for both _x and _y periods) ---
        for period in ['x', 'y']:
            for col in [f'current_balance_{period}', f'high_balance_{period}', f'credit_limit_{period}']:
                final_result[col] = pd.to_numeric(final_result[col], errors='coerce')
            
            final_result[f'lim_disbursed_{period}'] = np.nanmax(final_result[[f'high_balance_{period}', f'credit_limit_{period}']].values, axis=1)
            final_result[f'active_balance_{period}'] = np.where(final_result[f'Activity_Flag_{period}'] == 1, final_result[f'current_balance_{period}'], 0)
            final_result[f'utilisation_{period}'] = (final_result[f'current_balance_{period}'] / final_result[f'lim_disbursed_{period}']).replace([np.inf, -np.inf], np.nan).fillna(0)

            active_condition = final_result[f'Activity_Flag_{period}'] == 1
            final_result.loc[active_condition, f'total_lim_disbursed_{period}'] = final_result.loc[active_condition].groupby('customer_no')[f'lim_disbursed_{period}'].transform('sum')
            final_result.loc[active_condition, f'total_active_balance_{period}'] = final_result.loc[active_condition].groupby('customer_no')[f'active_balance_{period}'].transform('sum')
            
            cc_condition = (final_result[f'Activity_Flag_{period}'] == 1) & (final_result[f'priority_3_{period}'] == '01.0 CC')
            final_result.loc[cc_condition, f'total_cc_lim_disbursed_{period}'] = final_result.loc[cc_condition].groupby('customer_no')[f'lim_disbursed_{period}'].transform('sum')
            final_result.loc[cc_condition, f'total_cc_active_balance_{period}'] = final_result.loc[cc_condition].groupby('customer_no')[f'active_balance_{period}'].transform('sum')

            cols_to_fill = [
                f'total_lim_disbursed_{period}', f'total_active_balance_{period}',
                f'total_cc_lim_disbursed_{period}', f'total_cc_active_balance_{period}'
            ]
            final_result[cols_to_fill] = final_result.groupby('customer_no')[cols_to_fill].ffill().bfill()
            
            final_result[f'overall_utilisation_{period}'] = (final_result[f'total_active_balance_{period}'] / final_result[f'total_lim_disbursed_{period}']).replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # --- 💡 KEY FIX #2 IS HERE 💡 ---
            # Calculate CC util in a temporary series
            temp_cc_util = (final_result[f'total_cc_active_balance_{period}'] / final_result[f'total_cc_lim_disbursed_{period}']).replace([np.inf, -np.inf], np.nan)
            final_result[f'overall_cc_utilisation_{period}'] = temp_cc_util
            
            # Broadcast the single CC util value to all of that customer's rows
            final_result[f'overall_cc_utilisation_{period}'] = final_result.groupby('customer_no')[f'overall_cc_utilisation_{period}'].transform('max')
            
            # Fill remaining NaNs (for customers with no CCs) with 0
            final_result[f'overall_cc_utilisation_{period}'] = final_result[f'overall_cc_utilisation_{period}'].fillna(0)
            # --- End of Fix ---

        # --- 4. Difference & Coalesced Features ---
        final_result['utilisation_diff'] = final_result['utilisation_y'] - final_result['utilisation_x']
        final_result['overall_utilisation_diff'] = final_result['overall_utilisation_y'] - final_result['overall_utilisation_x']
        final_result['overall_cc_utilisation_diff'] = final_result['overall_cc_utilisation_y'] - final_result['overall_cc_utilisation_x']

        final_result['utilisation_percent_diff'] = (final_result['utilisation_diff'] / final_result['utilisation_x']).replace([np.inf, -np.inf], np.nan).fillna(0)
        final_result['overall_utilisation_percent_diff'] = (final_result['overall_utilisation_diff'] / final_result['overall_utilisation_x']).replace([np.inf, -np.inf], np.nan).fillna(0)
        final_result['overall_cc_utilisation_percent_diff'] = (final_result['overall_cc_utilisation_diff'] / final_result['overall_cc_utilisation_x']).replace([np.inf, -np.inf], np.nan).fillna(0)

        final_result = final_result.sort_values(by=['priority_3_y', 'date_opened'], ascending=[True, True])
        final_result['coalesced_priority'] = np.where(final_result['priority_3_y'].notnull(), final_result['priority_3_y'], final_result['priority_3_x'])
        final_result['coalesced_loan_type'] = np.where(final_result['loan_type_y'].notnull(), final_result['loan_type_y'], final_result['loan_type_x'])
        final_result['coalesced_open_date'] = pd.to_datetime(final_result['date_opened']).dt.strftime('%d %b, %Y').astype(str)

        # --- 5. Customer-level Aggregates & Flags ---
        final_result['risk_score_x'] = final_result.groupby('customer_no')['risk_score_x'].transform('max')
        final_result['risk_score_y'] = final_result.groupby('customer_no')['risk_score_y'].transform('max')
        final_result['risk_score_diff'] = final_result['risk_score_y'] - final_result['risk_score_x']
        
        final_result['total_active_accounts_y'] = final_result.groupby('customer_no')['Activity_Flag_y'].transform('sum')
        final_result['total_active_accounts_x'] = final_result.groupby('customer_no')['Activity_Flag_x'].transform('sum')

        final_result['total_active_cc_accounts_x'] = (final_result['priority_3_x'] == '01.0 CC') & (final_result['Activity_Flag_x'] == 1)
        final_result['total_active_cc_accounts_x'] = final_result.groupby('customer_no')['total_active_cc_accounts_x'].transform('sum')
        final_result['total_active_cc_accounts_y'] = (final_result['priority_3_y'] == '01.0 CC') & (final_result['Activity_Flag_y'] == 1)
        final_result['total_active_cc_accounts_y'] = final_result.groupby('customer_no')['total_active_cc_accounts_y'].transform('sum')
        
        final_result['rn'] = final_result.groupby('coalesced_loan_type')['date_opened'].rank(method='first').astype(int)
        final_result['new_account_flag'] = np.where((final_result['account_number_y'].notnull()) & (final_result['account_number_x'].isnull()) & (final_result['diff_sin_open_y'] <= 3), 1, 0)
        final_result['temp'] = np.where(final_result['latest_payment_dpd_status_y'] == 0, final_result['max_delinquency_latest_2_months_y'], final_result['latest_payment_dpd_status_y'])
        
        # --- 6. Final Cleanup ---
        final_result = final_result.apply(self._fill_nulls)
        
        return final_result