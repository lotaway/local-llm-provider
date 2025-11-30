-- Agent Context Storage Tables for PostgreSQL
-- This script creates the necessary tables for storing agent runtime context

-- Create agent_contexts table
CREATE TABLE IF NOT EXISTS agent_contexts (
    session_id VARCHAR(255) PRIMARY KEY,
    state_data JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index on updated_at for efficient cleanup queries
CREATE INDEX IF NOT EXISTS idx_agent_contexts_updated_at 
ON agent_contexts(updated_at);

-- Create index on created_at for analytics
CREATE INDEX IF NOT EXISTS idx_agent_contexts_created_at 
ON agent_contexts(created_at);

-- Optional: Create a function to automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Optional: Create trigger to auto-update updated_at
DROP TRIGGER IF EXISTS update_agent_contexts_updated_at ON agent_contexts;
CREATE TRIGGER update_agent_contexts_updated_at
    BEFORE UPDATE ON agent_contexts
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Optional: Create a function to cleanup old sessions
CREATE OR REPLACE FUNCTION cleanup_old_agent_contexts(days_to_keep INTEGER DEFAULT 7)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM agent_contexts 
    WHERE updated_at < CURRENT_TIMESTAMP - (days_to_keep || ' days')::INTERVAL;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Example usage of cleanup function:
-- SELECT cleanup_old_agent_contexts(7);  -- Clean up sessions older than 7 days

COMMENT ON TABLE agent_contexts IS 'Stores agent runtime context and state for session persistence';
COMMENT ON COLUMN agent_contexts.session_id IS 'Unique session identifier';
COMMENT ON COLUMN agent_contexts.state_data IS 'Serialized agent state as JSONB';
COMMENT ON COLUMN agent_contexts.created_at IS 'Timestamp when session was created';
COMMENT ON COLUMN agent_contexts.updated_at IS 'Timestamp when session was last updated';
