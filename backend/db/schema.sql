CREATE TABLE IF NOT EXISTS strategies (
    id SERIAL PRIMARY KEY,
    token_pair VARCHAR(50) NOT NULL,
    strategy_details JSONB NOT NULL,
    performance_metrics JSONB NOT NULL,
    timestamp TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS performance_metrics (
    id SERIAL PRIMARY KEY,
    token_pair VARCHAR(50) NOT NULL,
    agent_id VARCHAR(50) NOT NULL,
    metrics JSONB NOT NULL,
    timestamp TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS datasets (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    file_path TEXT NOT NULL,
    added_on TIMESTAMP DEFAULT NOW()
);
