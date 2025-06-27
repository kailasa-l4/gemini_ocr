#!/usr/bin/env python3
"""
Database setup script for OCR logging system
Creates all necessary tables and indexes
"""

import os
import sys
from dotenv import load_dotenv
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

def create_database_schema(database_url):
    """Create the database schema for OCR logging."""
    
    schema_sql = """
    -- OCR processing sessions (multiple sessions can run simultaneously)
    CREATE TABLE IF NOT EXISTS ocr_sessions (
        id SERIAL PRIMARY KEY,
        session_id VARCHAR(50) UNIQUE NOT NULL,
        hostname VARCHAR(100) NOT NULL,
        start_time TIMESTAMP DEFAULT NOW(),
        end_time TIMESTAMP,
        input_path VARCHAR(500),
        input_type VARCHAR(20),
        output_path VARCHAR(500),
        status VARCHAR(20) DEFAULT 'running',
        total_files INTEGER DEFAULT 0,
        completed_files INTEGER DEFAULT 0,
        failed_files INTEGER DEFAULT 0,
        configuration JSONB,
        created_at TIMESTAMP DEFAULT NOW()
    );
    
    -- Individual file processing logs
    CREATE TABLE IF NOT EXISTS processing_logs (
        id SERIAL PRIMARY KEY,
        session_id VARCHAR(50) NOT NULL,
        file_path VARCHAR(500) NOT NULL,
        page_number INTEGER,
        processing_start TIMESTAMP DEFAULT NOW(),
        processing_end TIMESTAMP,
        status VARCHAR(30),
        legibility_score DECIMAL(3,2),
        semantic_score DECIMAL(3,2),
        ocr_confidence DECIMAL(3,2),
        processing_time DECIMAL(10,6),
        text_clarity VARCHAR(20),
        image_quality VARCHAR(20),
        ocr_prediction VARCHAR(30),
        semantic_prediction VARCHAR(30),
        visible_text_sample TEXT,
        language_detected VARCHAR(50),
        issues_found TEXT,
        error_message TEXT,
        created_at TIMESTAMP DEFAULT NOW()
    );
    
    -- System errors and exceptions
    CREATE TABLE IF NOT EXISTS error_logs (
        id SERIAL PRIMARY KEY,
        session_id VARCHAR(50),
        error_type VARCHAR(100),
        error_message TEXT,
        stack_trace TEXT,
        file_path VARCHAR(500),
        function_name VARCHAR(100),
        line_number INTEGER,
        severity VARCHAR(20) DEFAULT 'medium',
        hostname VARCHAR(100),
        created_at TIMESTAMP DEFAULT NOW()
    );
    
    -- Create indexes for better performance
    CREATE INDEX IF NOT EXISTS idx_ocr_sessions_session_id ON ocr_sessions(session_id);
    CREATE INDEX IF NOT EXISTS idx_ocr_sessions_hostname ON ocr_sessions(hostname);
    CREATE INDEX IF NOT EXISTS idx_ocr_sessions_status ON ocr_sessions(status);
    CREATE INDEX IF NOT EXISTS idx_ocr_sessions_start_time ON ocr_sessions(start_time);
    
    CREATE INDEX IF NOT EXISTS idx_processing_logs_session_id ON processing_logs(session_id);
    CREATE INDEX IF NOT EXISTS idx_processing_logs_status ON processing_logs(status);
    CREATE INDEX IF NOT EXISTS idx_processing_logs_file_path ON processing_logs(file_path);
    CREATE INDEX IF NOT EXISTS idx_processing_logs_created_at ON processing_logs(created_at);
    
    CREATE INDEX IF NOT EXISTS idx_error_logs_session_id ON error_logs(session_id);
    CREATE INDEX IF NOT EXISTS idx_error_logs_severity ON error_logs(severity);
    CREATE INDEX IF NOT EXISTS idx_error_logs_error_type ON error_logs(error_type);
    CREATE INDEX IF NOT EXISTS idx_error_logs_created_at ON error_logs(created_at);
    """
    
    print(f"Connecting to database: {database_url}")
    
    try:
        # Connect to database
        conn = psycopg2.connect(database_url)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        print("✓ Connected to database successfully")
        
        # Execute schema creation
        print("Creating database schema...")
        cursor.execute(schema_sql)
        print("✓ Database schema created successfully")
        
        # Verify tables were created
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('ocr_sessions', 'processing_logs', 'error_logs')
            ORDER BY table_name;
        """)
        
        tables = cursor.fetchall()
        print(f"✓ Created tables: {[table[0] for table in tables]}")
        
        # Check indexes
        cursor.execute("""
            SELECT indexname 
            FROM pg_indexes 
            WHERE tablename IN ('ocr_sessions', 'processing_logs', 'error_logs')
            AND indexname LIKE 'idx_%'
            ORDER BY indexname;
        """)
        
        indexes = cursor.fetchall()
        print(f"✓ Created indexes: {len(indexes)} performance indexes")
        
        cursor.close()
        conn.close()
        
        print("\n🎉 Database setup completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Database setup failed: {e}")
        return False

def main():
    """Main function to set up the database."""
    print("=== OCR Database Setup ===\n")
    
    # Load environment variables
    load_dotenv()
    
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        print("Error: DATABASE_URL not found in environment")
        print("Please set DATABASE_URL in your .env file:")
        print("DATABASE_URL=postgresql://ocr:rmtetn5qek6zikol@157.180.15.165:1111/db")
        return False
    
    # Create database schema
    success = create_database_schema(database_url)
    
    if success:
        print("\nNext steps:")
        print("1. Run the test script: python test_db_integration.py")
        print("2. Start using OCR with database logging enabled")
        return True
    else:
        print("\nPlease check your database connection and try again.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)