# customer_analyzer.py
# io.StringIO(): Creates an empty, temporary text box in your computer's memory.
# redirect_stdout(): Tells the print() function to write everything into that text box instead of showing it on your screen.
import pandas as pd
import io
from contextlib import redirect_stdout

class CustomerScoreAnalyzer:
    """
    Prepares data for LLM fine-tuning by separating a customer's credit profile
    into a detailed 'customer_info' (context) and a narrative 'customer_credit_update' (target).
    """

    def _generate_info_report(self, final_result1, df_enq, writer):
        """Generates the comprehensive 'customer_info' fact sheet."""
        with redirect_stdout(writer):
            user_id = final_result1['customer_no'].iloc[0]
            print(f"--- Credit Profile Report for Customer: {user_id} ---")

            # --- High-Level Summaries (Before & After) ---
            print("\n## Key Metric Summary")
            latest_record = final_result1.iloc[0]
            print(f"-  Risk Score : {latest_record.get('risk_score_y', 'N/A')} (was {latest_record.get('risk_score_x', 'N/A')})")
            print(f"-  Overall Utilization : {latest_record.get('overall_utilisation_y', 0):.2%} (was {latest_record.get('overall_utilisation_x', 0):.2%})")
            if latest_record.get('total_active_cc_accounts_y', 0) > 0 or latest_record.get('total_active_cc_accounts_x', 0) > 0:
                print(f"-  Credit Card Utilization : {latest_record.get('overall_cc_utilisation_y', 0):.2%} (was {latest_record.get('overall_cc_utilisation_x', 0):.2%})")
            print(f"-  Total Active Accounts : {latest_record.get('total_active_accounts_y', 'N/A')} (was {latest_record.get('total_active_accounts_x', 'N/A')})")

            # --- Credit Mix Information ---
            print("\n## Current Credit Mix")
            if 'secured_unsecured_y' in final_result1.columns:
                total_loans = len(final_result1[final_result1['account_number_y'] != 'NA'])
                total_active_loans = final_result1['Activity_Flag_y'].sum()
                print(f"-  Total Accounts : {total_loans} ({total_active_loans} active)")
                active_loan_counts = final_result1[final_result1['Activity_Flag_y'] == 1]['secured_unsecured_y'].value_counts()
                print(f"-  Active Secured Products : {active_loan_counts.get('1. Secured', 0)}")
                print(f"-  Active Unsecured Products : {active_loan_counts.get('2. Unsecured', 0)}")

            if 'lender_type' in final_result1.columns:
                loan_counts = final_result1[final_result1['Activity_Flag_y'] == 1]['lender_type'].value_counts()
                print("-  Active Lender Distribution : "
                      f"Public({loan_counts.get('Public sector', 0)}), "
                      f"Private({loan_counts.get('Private sector', 0)}), "
                      f"NBFC({loan_counts.get('NBFC', 0)}), "
                      f"Corporate({loan_counts.get('Corporate bank', 0)}), "
                      f"Foreign({loan_counts.get('Foreign bank', 0)})")


            # --- Detailed Account-Level Breakdown ---
            print("\n## Account Details Breakdown")
            for _, row in final_result1.iterrows():
                status = "Active"
                if row.get('_merge') == 'left_only':
                    status = "Removed from Report"
                elif row.get('new_account_flag') == 1:
                    status = "New Account"
                elif row.get('Activity_Flag_x') == 1 and row.get('Activity_Flag_y') == 0:
                    status = "Closed this Period"

                print(f"\n-  Account : {row.get('creditor_name', 'N/A')} - {row.get('coalesced_loan_type', 'N/A')} ({row.get('acc_no', 'N/A')})")
                print(f"  -  Status : {status}")
                print(f"  -  DPD : {row.get('latest_payment_dpd_status_y', 'N/A')} days (was {row.get('latest_payment_dpd_status_x', 'N/A')} days)")
                if 'CC' in str(row.get('coalesced_loan_type', '')).upper() or row.get('utilisation_x', 0) > 0 or row.get('utilisation_y', 0) > 0:
                     print(f"  -  Utilization : {row.get('utilisation_y', 0):.2%} (was {row.get('utilisation_x', 0):.2%})")
                if row.get('account_type_symbol_x') != row.get('account_type_symbol_y'):
                    print(f"  -  Info Change : Account type is now '{row.get('account_type_symbol_y')}' (was '{row.get('account_type_symbol_x')}')")

            # --- Recent Enquiries ---
            if df_enq is not None and not df_enq.empty:
                print("\n## Recent Credit Enquiries")
                for _, row in df_enq.iterrows():
                    print(f"-  Lender : {row.get('subscriber_name', 'N/A')},  Type : {row.get('loan_type', 'N/A')},  Date : {row.get('inquiry_date', 'N/A')}")

    def _generate_update_narrative(self, final_result1, df_enq, writer):
        """Generates the 'customer_credit_update' narrative of Good/Bad changes."""
        with redirect_stdout(writer):
            user_id = final_result1['customer_no'].iloc[0]

            # --- SCORE UPDATES ---
            if 'risk_score_diff' in final_result1.columns and not final_result1['risk_score_diff'].dropna().empty:
                if final_result1['risk_score_diff'].max() < 0:
                    drop = -1 * final_result1['risk_score_diff'].iloc[0]
                    print(f"Bad:- User's score has reduced between 2 months by {drop} points.")
                if final_result1['risk_score_diff'].max() > 0:
                    increase = final_result1['risk_score_diff'].iloc[0]
                    print(f"Good:- User's score has increased between 2 months by {increase} points.")

            # --- "BAD" UPDATES ---

            # Delinquency
            delinquent_now = final_result1[(final_result1['latest_payment_dpd_status_y'] > 0) & (final_result1['Activity_Flag_y'] == 1)]
            if not delinquent_now.empty:
                msg = ', '.join([f"{row['creditor_name']} {row['loan_type_y']} ({row['temp']} days)" for _, row in delinquent_now.iterrows()])
                print(f"Bad:- User is delinquent on accounts: {msg}.")

            freshly_delinquent = final_result1[(final_result1.get('temp', 0) > 0) & (final_result1['Activity_Flag_y'] == 1) & (final_result1.get('max_dpd_l2m_x', 0) <= 0) & (final_result1.get('max_dpd_l3m_x', 0) <= 0) & (final_result1.get('max_dpd_l2m_y', 0) > 0)]
            if not freshly_delinquent.empty:
                msg = ', '.join([f"{row['creditor_name']} {row['loan_type_y']} ({row['temp']} days)" for _, row in freshly_delinquent.iterrows()])
                print(f"Bad:- User has become freshly delinquent on accounts: {msg}.")

            # Utilization
            if 'utilisation_y' in final_result1.columns and not final_result1['utilisation_y'].dropna().empty and final_result1['utilisation_y'].max() <= 0:
                print(f"Bad:- User has become dormant and has zero overall credit utilization.")

            if 'overall_utilisation_percent_diff' in final_result1.columns:
                overall_util_change = final_result1['overall_utilisation_percent_diff'].max() * 100
                cc_util_change = final_result1.get('overall_cc_utilisation_percent_diff', pd.Series([0])).max() * 100
                if overall_util_change > 0:
                    print(f"Bad:- User's overall utilisation has increased by {overall_util_change:.2f} percentage points.")
                if cc_util_change > 0:
                    print(f"Bad:- User's cc utilisation has increased by {cc_util_change:.2f} percentage points.")

            util_increase = final_result1[(final_result1.get('utilisation_percent_diff', 0) > 0.5) & (final_result1['Activity_Flag_y'] == 1)]
            if not util_increase.empty:
                msg = ', '.join([f"{row['creditor_name']} ({row['utilisation_percent_diff']*100:.0f}%)" for _, row in util_increase.iterrows()])
                print(f"Bad:- User has increased utilisation on following accounts: {msg}.")

            if 'overall_cc_utilisation_y' in final_result1.columns:
                util_all_y = final_result1['overall_utilisation_y'].max() * 100
                util_cc_y = final_result1['overall_cc_utilisation_y'].max() * 100
                if util_all_y >= 30:
                    print(f"Bad:- User's overall utilisation is high at {util_all_y:.2f}%.")
                if util_cc_y >= 30:
                    print(f"Bad:- User's cc utilisation is high at {util_cc_y:.2f}%.")

            high_util_accounts = final_result1[(final_result1.get('utilisation_y', 0) >= 0.3) & (final_result1['Activity_Flag_y'] == 1)]
            if not high_util_accounts.empty:
                msg = ', '.join([f"{row['creditor_name']} ({row['utilisation_y']*100:.0f}%)" for _, row in high_util_accounts.iterrows()])
                print(f"Bad:- User has high utilisation (>30%) in following accounts: {msg}.")

            # Account Activity
            new_accounts_bad = final_result1[(final_result1.get('new_account_flag') == 1) & (final_result1['rn'] != 1)]
            if not new_accounts_bad.empty:
                msg = ', '.join([f"{row['creditor_name']} ({row['loan_type_y']})" for _, row in new_accounts_bad.iterrows()])
                print(f"Bad:- User has opened new following accounts: {msg}.")

            reporting_errors = final_result1[(final_result1['account_type_symbol_y'] != final_result1['account_type_symbol_x']) & (final_result1['Activity_Flag_y'] == 1) & (final_result1['Activity_Flag_x'] == 1)]
            if not reporting_errors.empty:
                msg = ', '.join([f"{row['creditor_name']} (from {row['account_type_symbol_x']} to {row['account_type_symbol_y']})" for _, row in reporting_errors.iterrows()])
                print(f"Bad:- User's following accounts were reported wrongly: {msg}.")

            if df_enq is not None and not df_enq.empty:
                msg = ', '.join([f"{row['subscriber_name']} ({row['loan_type']})" for _, row in df_enq.iterrows()])
                print(f"Bad:- User has made new inquiries with the following lenders: {msg}.")


            # --- "GOOD" UPDATES ---

            # Delinquency
            delinquency_reduced = final_result1[(final_result1.get('latest_payment_dpd_status_diff', 0) < -1) & (final_result1['Activity_Flag_x'] == 1) & (final_result1.get('max_dpd_l2m_x', 0) > 0)]
            if not delinquency_reduced.empty:
                msg = ', '.join([f"{row['creditor_name']} (by {abs(row['latest_payment_dpd_status_diff'])} days)" for _, row in delinquency_reduced.iterrows()])
                print(f"Good:- User's delinquency has reduced in the following accounts: {msg}.")

            not_delinquent_anymore = final_result1[(final_result1.get('latest_payment_dpd_status_diff', 0) < -1) & (final_result1['latest_payment_dpd_status_y'] == 0) & (final_result1.get('max_dpd_l2m_x', 0) > 0) & (final_result1.get('max_dpd_l3m_x', 0) > 0) & (final_result1['Activity_Flag_x'] == 1)]
            if not not_delinquent_anymore.empty:
                msg = ', '.join([f"{row['creditor_name']} ({row['loan_type_y']})" for _, row in not_delinquent_anymore.iterrows()])
                print(f"Good:- User is no more delinquent on the following accounts: {msg}.")

            # Utilization
            if 'overall_utilisation_percent_diff' in final_result1.columns:
                overall_util_change = final_result1['overall_utilisation_percent_diff'].max() * 100
                cc_util_change = final_result1.get('overall_cc_utilisation_percent_diff', pd.Series([0])).max() * 100
                if overall_util_change < 0:
                    print(f"Good:- User's overall utilisation has decreased by {abs(overall_util_change):.2f} percentage points.")
                if cc_util_change < 0:
                    print(f"Good:- User's cc utilisation has decreased by {abs(cc_util_change):.2f} percentage points.")
            
            if 'overall_cc_utilisation_y' in final_result1.columns:
                util_all_y = final_result1['overall_utilisation_y'].max() * 100
                util_cc_y = final_result1['overall_cc_utilisation_y'].max() * 100
                if util_all_y < 30:
                    print(f"Good:- User's overall utilisation is healthy at {util_all_y:.2f}%.")
                if util_cc_y < 30:
                    print(f"Good:- User's cc utilisation is healthy at {util_cc_y:.2f}%.")

            util_reduced = final_result1[(final_result1.get('utilisation_percent_diff', 0) < -0.1) & (final_result1['Activity_Flag_x'] == 1)]
            if not util_reduced.empty:
                msg = ', '.join([f"{row['creditor_name']} ({abs(row['utilisation_percent_diff']*100):.0f}%)" for _, row in util_reduced.iterrows()])
                print(f"Good:- User has reduced their utilisation in the following accounts: {msg}.")

            low_util_accounts = final_result1[(final_result1.get('utilisation_y', 0) < 0.3) & (final_result1['Activity_Flag_y'] == 1)]
            if not low_util_accounts.empty:
                msg = ', '.join([f"{row['creditor_name']} ({row['utilisation_y']*100:.0f}%)" for _, row in low_util_accounts.iterrows()])
                print(f"Good:- User has utilisation less than 30% in the following accounts: {msg}.")

            # Account Activity
            fixed_reporting = final_result1[final_result1['_merge'] == 'left_only']
            if not fixed_reporting.empty:
                msg = ', '.join([f"{row['creditor_name']} ({row['coalesced_loan_type']})" for _, row in fixed_reporting.iterrows()])
                print(f"Good:- User's following accounts were removed from their report: {msg}.")

            account_closed = final_result1[(final_result1.get('Activity_Flag_y') == 0) & (final_result1.get('Activity_Flag_x') == 1)]
            if not account_closed.empty:
                msg = ', '.join([f"{row['creditor_name']} ({row['loan_type_y']})" for _, row in account_closed.iterrows()])
                print(f"Good:- User has closed the following accounts: {msg}.")

            new_accounts_good = final_result1[(final_result1.get('new_account_flag') == 1) & (final_result1['rn'] == 1)]
            if not new_accounts_good.empty:
                msg = ', '.join([f"{row['creditor_name']} ({row['loan_type_y']})" for _, row in new_accounts_good.iterrows()])
                print(f"Good:- User has opened new following accounts: {msg}.")

            new_after_dormancy = final_result1[(final_result1.get('new_account_flag') == 1) & (final_result1.get('total_active_accounts_y', 0) >= 1) & (final_result1.get('total_active_accounts_x', 0) == 0)]
            if not new_after_dormancy.empty:
                msg = ', '.join([f"{row['creditor_name']} ({row['loan_type_y']})" for _, row in new_after_dormancy.iterrows()])
                print(f"Good:- User has opened new following accounts after a period of dormancy: {msg}.")


    def generate_training_data(self, final_result1, df_enq=None):
        """
        Processes raw data to generate a fine-tuning ready DataFrame.
        """
        if 'customer_no' not in final_result1.columns:
            raise ValueError("The input DataFrame must contain a 'customer_no' column.")
            
        training_data = {}
        for customer_id, customer_group in final_result1.groupby('customer_no'):
            info_buffer = io.StringIO()
            update_buffer = io.StringIO()
            
            customer_enq_df = None
            if df_enq is not None and 'customer_no' in df_enq.columns:
                customer_enq_df = df_enq[df_enq['customer_no'] == customer_id].copy()
            
            self._generate_info_report(customer_group, customer_enq_df, info_buffer)
            self._generate_update_narrative(customer_group, customer_enq_df, update_buffer) 
            
            training_data[customer_id] = (info_buffer.getvalue(), update_buffer.getvalue())
            
        training_df = pd.DataFrame.from_dict(
            training_data, orient='index', 
            columns=['customer_info', 'customer_credit_update']
        ).reset_index().rename(columns={'index': 'customer_no'})
        
        return training_df