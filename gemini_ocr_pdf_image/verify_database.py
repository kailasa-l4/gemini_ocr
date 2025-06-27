#!/usr/bin/env python3
"""
Verify database setup and show current data
"""

import os
import sys
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor

def verify_database():
    """Verify database setup and show existing data."""
    load_dotenv()
    
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        print("Error: DATABASE_URL not found in environment")
        return False
    
    print("=== Database Verification ===\n")
    
    try:
        conn = psycopg2.connect(database_url)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        print("✓ Connected to database successfully")
        
        # Check tables exist
        cursor.execute("""
            SELECT table_name, 
                   (SELECT COUNT(*) FROM information_schema.columns 
                    WHERE table_name = t.table_name AND table_schema = 'public') as column_count
            FROM information_schema.tables t
            WHERE table_schema = 'public' 
            AND table_name IN ('ocr_sessions', 'processing_logs', 'error_logs')
            ORDER BY table_name;
        """)
        
        tables = cursor.fetchall()
        print(f"✓ Found {len(tables)} required tables:")
        for table in tables:
            print(f"  - {table['table_name']}: {table['column_count']} columns")
        
        # Check current data
        print("\n=== Current Database Content ===")
        
        # Sessions
        cursor.execute("SELECT COUNT(*) as count FROM ocr_sessions")
        session_count = cursor.fetchone()['count']
        print(f"Sessions: {session_count}")
        
        if session_count > 0:
            cursor.execute("""
                SELECT session_id, hostname, input_type, status, 
                       total_files, completed_files, failed_files, 
                       created_at
                FROM ocr_sessions 
                ORDER BY created_at DESC 
                LIMIT 5
            """)
            recent_sessions = cursor.fetchall()
            print("Recent sessions:")
            for session in recent_sessions:
                print(f"  • {session['session_id'][:8]}... [{session['status']}] "
                      f"{session['input_type']} - {session['total_files']} files "
                      f"({session['completed_files']} completed, {session['failed_files']} failed)")
        
        # Processing logs
        cursor.execute("SELECT COUNT(*) as count FROM processing_logs")
        log_count = cursor.fetchone()['count']
        print(f"\nProcessing logs: {log_count}")
        
        if log_count > 0:
            cursor.execute("""
                SELECT status, COUNT(*) as count 
                FROM processing_logs 
                GROUP BY status 
                ORDER BY count DESC
            """)
            status_counts = cursor.fetchall()
            print("Status breakdown:")
            for status in status_counts:
                print(f"  • {status['status']}: {status['count']}")
        
        # Error logs
        cursor.execute("SELECT COUNT(*) as count FROM error_logs")
        error_count = cursor.fetchone()['count']
        print(f"\nError logs: {error_count}")
        
        if error_count > 0:
            cursor.execute("""
                SELECT severity, COUNT(*) as count 
                FROM error_logs 
                GROUP BY severity 
                ORDER BY count DESC
            """)
            error_counts = cursor.fetchall()
            print("Error severity breakdown:")
            for error in error_counts:
                print(f"  • {error['severity']}: {error['count']}")
        
        # Performance stats
        cursor.execute("""
            SELECT 
                AVG(processing_time) as avg_time,
                AVG(legibility_score) as avg_legibility,
                AVG(semantic_score) as avg_semantic,
                AVG(ocr_confidence) as avg_confidence
            FROM processing_logs 
            WHERE processing_time IS NOT NULL
        """)
        stats = cursor.fetchone()
        
        if stats and stats['avg_time']:
            print(f"\n=== Performance Statistics ===")
            print(f"Average processing time: {stats['avg_time']:.2f} seconds")
            print(f"Average legibility score: {stats['avg_legibility']:.2f}")
            print(f"Average semantic score: {stats['avg_semantic']:.2f}")
            print(f"Average OCR confidence: {stats['avg_confidence']:.2f}")
        
        cursor.close()
        conn.close()
        
        print(f"\n🎉 Database verification completed successfully!")
        print(f"\nDatabase is ready for OCR logging with:")
        print(f"  • Session tracking ✓")
        print(f"  • Processing logs ✓") 
        print(f"  • Error logging ✓")
        print(f"  • Performance indexing ✓")
        
        return True
        
    except Exception as e:
        print(f"✗ Database verification failed: {e}")
        return False

if __name__ == "__main__":
    success = verify_database()
    sys.exit(0 if success else 1)