import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_synthetic_data(num_users=200, output_dir="data"):
    """
    Generates synthetic transaction logs and downstream evaluation labels.
    """
    os.makedirs(output_dir, exist_ok=True)
    random.seed(42)
    np.random.seed(42)
    
    # MCCs (Merchant Category Codes) representing different merchant groups
    mcc_list = [5411, 5812, 5541, 5311, 5732, 5912, 4814, 5814, 6011, 7997]
    # Risk weights for different MCCs
    mcc_risk_weights = {
        5411: 0.1,  # Groceries (low risk)
        5812: 0.3,  # Dining (medium risk)
        5541: 0.2,  # Gas (low risk)
        5311: 0.2,  # Dept stores (low risk)
        5732: 0.8,  # Electronics (high risk for fraud/default)
        5912: 0.2,  # Drug stores (low risk)
        4814: 0.4,  # Telecom (medium risk)
        5814: 0.3,  # Fast food (low risk)
        6011: 0.9,  # ATM cash withdrawal (high risk)
        7997: 0.5   # Gyms/Recreation (medium risk)
    }

    transactions = []
    user_labels = []
    
    start_date = datetime(2026, 1, 1)

    for user_id in range(num_users):
        # Determine user latent risk factors
        latent_risk = np.random.beta(2, 5) # Beta distribution: most users are low risk, some high risk
        
        # User characteristics
        base_default_prob = latent_risk * 0.7 + np.random.normal(0, 0.05)
        base_default_prob = np.clip(base_default_prob, 0.0, 1.0)
        
        # Number of transactions for this user
        num_tx = random.randint(15, 60)
        current_time = start_date + timedelta(days=random.randint(0, 30))
        
        user_tx_mccs = []
        user_tx_amounts = []
        
        for tx_idx in range(num_tx):
            # Select MCC based on user risk (high-risk users do more cash outs & electronics)
            if latent_risk > 0.5 and random.random() < 0.6:
                mcc = random.choice([6011, 5732]) # cash / electronics
            else:
                mcc = random.choice(mcc_list)
                
            # Transaction amount: log-normal distribution
            base_amount = 20.0
            if mcc == 6011:
                base_amount = 100.0  # Cash withdrawals are larger
            elif mcc == 5732:
                base_amount = 250.0  # Electronics are larger
                
            amount = np.random.lognormal(mean=np.log(base_amount), sigma=0.6)
            amount = round(float(amount), 2)
            
            # Time delta between transactions
            current_time += timedelta(hours=random.randint(1, 72))
            
            # Label transaction fraud
            # Fraud is likely if transaction is high amount and high-risk MCC, amplified by user latent risk
            mcc_risk = mcc_risk_weights.get(mcc, 0.3)
            fraud_prob = 0.01 + 0.15 * mcc_risk * (amount > 150) + 0.1 * latent_risk
            is_fraud = 1 if random.random() < fraud_prob else 0
            
            transactions.append({
                "transaction_id": f"tx_{user_id}_{tx_idx}",
                "user_id": user_id,
                "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "mcc": mcc,
                "amount": amount,
                "is_fraud": is_fraud
            })
            
            user_tx_mccs.append(mcc)
            user_tx_amounts.append(amount)

        # Downstream Credit Default Label (user level)
        # Default is correlated with high cash out ratio, high variance in amounts, and latent risk
        cash_out_ratio = sum(1 for m in user_tx_mccs if m == 6011) / len(user_tx_mccs)
        default_score = base_default_prob * 0.5 + cash_out_ratio * 0.4 + (np.std(user_tx_amounts) / 200.0) * 0.1
        default_prob = np.clip(default_score, 0.01, 0.99)
        default_label = 1 if random.random() < default_prob else 0
        
        user_labels.append({
            "user_id": user_id,
            "credit_default": default_label,
            "latent_risk": latent_risk # useful for validation, not used directly by models
        })

    # Save to files
    tx_df = pd.DataFrame(transactions)
    user_df = pd.DataFrame(user_labels)
    
    tx_df.to_csv(os.path.join(output_dir, "transactions.csv"), index=False)
    user_df.to_csv(os.path.join(output_dir, "users.csv"), index=False)
    
    print(f"Generated {len(tx_df)} transactions for {len(user_df)} users.")
    print(f"Credit default rate: {user_df['credit_default'].mean():.2%}")
    print(f"Transaction fraud rate: {tx_df['is_fraud'].mean():.2%}")

if __name__ == "__main__":
    generate_synthetic_data()
