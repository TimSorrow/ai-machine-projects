import os
import random
import numpy as np
import pandas as pd

def generate_behavioral_data(num_users=200, output_dir="data"):
    """
    Generates synthetic clickstream behavioral sequences and LTV metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    random.seed(42)
    np.random.seed(42)
    
    events_map = {
        0: "PAD",
        1: "login",
        2: "view_balance",
        3: "view_limits",
        4: "apply_for_card",
        5: "activate_card",
        6: "pay_bill",
        7: "support_chat",
        8: "increase_limit_request",
        9: "logout"
    }

    # Transition probability matrix (simplified state transitions)
    # Rows represent current state (1-9), columns represent next state probability
    # e.g., state 1 (login) is likely followed by 2 (view_balance) or 3 (view_limits)
    transitions = {
        1: [0.0, 0.0, 0.4, 0.3, 0.1, 0.0, 0.1, 0.05, 0.0, 0.05], # login -> ...
        2: [0.0, 0.0, 0.1, 0.1, 0.1, 0.0, 0.3, 0.1, 0.1, 0.2],   # view_balance -> ...
        3: [0.0, 0.0, 0.2, 0.1, 0.3, 0.05, 0.1, 0.05, 0.15, 0.05],# view_limits -> ...
        4: [0.0, 0.0, 0.1, 0.1, 0.0, 0.4, 0.1, 0.1, 0.1, 0.1],   # apply_for_card -> ...
        5: [0.0, 0.0, 0.2, 0.1, 0.0, 0.0, 0.4, 0.1, 0.1, 0.1],   # activate_card -> ...
        6: [0.0, 0.0, 0.3, 0.1, 0.1, 0.0, 0.2, 0.1, 0.1, 0.1],   # pay_bill -> ...
        7: [0.0, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.3, 0.1, 0.1],   # support_chat -> ...
        8: [0.0, 0.0, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.0, 0.1],   # increase_limit -> ...
        9: [0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]    # logout -> relogin or exit
    }

    user_events = []
    user_ltvs = []

    for user_id in range(num_users):
        # User characteristics affecting sequence length and value
        loyalty = np.random.beta(3, 3) # Beta distribution for engagement level
        
        seq = [1] # start with login
        current_state = 1
        
        # Determine sequence length based on loyalty
        max_len = int(10 + loyalty * 40)
        
        for _ in range(max_len):
            probs = transitions.get(current_state, transitions[1])
            # Sample next state
            next_state = np.random.choice(list(events_map.keys()), p=probs)
            
            if next_state == 9: # logout
                seq.append(next_state)
                # Decide if session ends or starts another login sequence
                if random.random() > 0.4:
                    break
                else:
                    seq.append(1)
                    current_state = 1
            else:
                seq.append(next_state)
                current_state = next_state

        # Compute Customer Lifetime Value (LTV)
        # LTV is high if user activates cards, pays bills, and request limits.
        # Support chats slightly penalize or correlate with lower LTV.
        num_cards = sum(1 for e in seq if e == 4 or e == 5)
        num_payments = sum(1 for e in seq if e == 6)
        num_support = sum(1 for e in seq if e == 7)
        num_limits = sum(1 for e in seq if e == 8)
        
        # Base LTV is determined by loyalty and transaction patterns
        ltv_score = 50.0 + (num_cards * 120.0) + (num_payments * 45.0) + (num_limits * 60.0) - (num_support * 15.0)
        ltv_val = ltv_score * (1.0 + np.random.normal(0, 0.1))
        ltv_val = max(10.0, round(float(ltv_val), 2))
        
        user_events.append({
            "user_id": user_id,
            "events": ",".join(map(str, seq))
        })
        
        user_ltvs.append({
            "user_id": user_id,
            "ltv": ltv_val
        })

    # Save to files
    events_df = pd.DataFrame(user_events)
    ltv_df = pd.DataFrame(user_ltvs)
    
    events_df.to_csv(os.path.join(output_dir, "clickstream_events.csv"), index=False)
    ltv_df.to_csv(os.path.join(output_dir, "clickstream_users.csv"), index=False)
    
    print(f"Generated clickstream logs for {len(events_df)} users.")
    print(f"Average sequence length: {events_df['events'].str.count(',').mean() + 1:.1f}")
    print(f"Average user LTV: ${ltv_df['ltv'].mean():.2f}")

if __name__ == "__main__":
    generate_behavioral_data()
