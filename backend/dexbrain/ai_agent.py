from dexbrain.db_manager import DexBrainDB

class DexterAgent:
    def __init__(self):
        self.db = DexBrainDB()

    def suggest_strategy(self, token_pair, user_risk):
        strategy = self.db.query_strategies(token_pair)
        if strategy:
            print("Using historical data for suggestion.")
            return strategy
        else:
            print("No historical data found. Fetching live data...")
            # Fetch and analyze live data logic here
            return {"range": (0.95, 1.05), "risk": user_risk}
