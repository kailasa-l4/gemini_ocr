#!/usr/bin/env python3
"""
Check exactly what tables exist in the database
"""

import os
import sys
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor

def check_all_tables():
    """Check all tables in the database."""
    load_dotenv()
    
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        print("Error: DATABASE_URL not found in environment")
        return False
    
    print("=== Complete Database Table Check ===\n")
    
    try:
        conn = psycopg2.connect(database_url)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Get ALL tables in the database
        cursor.execute("""
            SELECT 
                table_name,
                table_schema,
                table_type
            FROM information_schema.tables 
            WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
            ORDER BY table_schema, table_name;
        """)
        
        all_tables = cursor.fetchall()
        print(f"Total tables found: {len(all_tables)}")
        print("\nAll tables in database:")
        for table in all_tables:
            print(f"  • {table['table_schema']}.{table['table_name']} ({table['table_type']})")
        
        # Check specifically for our OCR tables
        expected_tables = ['ocr_sessions', 'processing_logs', 'error_logs']
        
        print(f"\n=== OCR Tables Check ===")
        print(f"Expected OCR tables: {len(expected_tables)}")
        
        found_ocr_tables = []
        for table in all_tables:
            if table['table_name'] in expected_tables:
                found_ocr_tables.append(table['table_name'])
        
        print(f"Found OCR tables: {len(found_ocr_tables)}")
        
        for table_name in expected_tables:
            if table_name in found_ocr_tables:
                print(f"  ✓ {table_name}")
                
                # Get column details for each table
                cursor.execute("""
                    SELECT 
                        column_name,
                        data_type,
                        is_nullable,
                        column_default
                    FROM information_schema.columns 
                    WHERE table_name = %s AND table_schema = 'public'
                    ORDER BY ordinal_position;
                """, (table_name,))
                
                columns = cursor.fetchall()
                print(f"    Columns: {len(columns)}")
                for col in columns:
                    nullable = "NULL" if col['is_nullable'] == 'YES' else "NOT NULL"
                    default = f" DEFAULT {col['column_default']}" if col['column_default'] else ""
                    print(f"      - {col['column_name']}: {col['data_type']} {nullable}{default}")
            else:
                print(f"  ✗ {table_name} - MISSING!")
        
        # Check indexes
        print(f"\n=== Indexes Check ===")
        cursor.execute("""
            SELECT 
                indexname,
                tablename,
                indexdef
            FROM pg_indexes 
            WHERE tablename IN ('ocr_sessions', 'processing_logs', 'error_logs')
            AND indexname LIKE 'idx_%'
            ORDER BY tablename, indexname;
        """)
        
        indexes = cursor.fetchall()
        print(f"OCR-related indexes found: {len(indexes)}")
        
        current_table = None
        for idx in indexes:
            if idx['tablename'] != current_table:
                current_table = idx['tablename']
                print(f"\n  {current_table}:")
            print(f"    • {idx['indexname']}")
        
        # Check constraints and foreign keys
        print(f"\n=== Constraints Check ===")
        cursor.execute("""
            SELECT 
                tc.constraint_name,
                tc.table_name,
                tc.constraint_type,
                kcu.column_name,
                ccu.table_name AS foreign_table_name,
                ccu.column_name AS foreign_column_name
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
              ON tc.constraint_name = kcu.constraint_name
              AND tc.table_schema = kcu.table_schema
            LEFT JOIN information_schema.constraint_column_usage AS ccu
              ON ccu.constraint_name = tc.constraint_name
              AND ccu.table_schema = tc.table_schema
            WHERE tc.table_name IN ('ocr_sessions', 'processing_logs', 'error_logs')
            ORDER BY tc.table_name, tc.constraint_type, tc.constraint_name;
        """)
        
        constraints = cursor.fetchall()
        print(f"Constraints found: {len(constraints)}")
        
        current_table = None
        for constraint in constraints:
            if constraint['table_name'] != current_table:
                current_table = constraint['table_name']
                print(f"\n  {current_table}:")
            
            constraint_info = f"    • {constraint['constraint_name']} ({constraint['constraint_type']})"
            if constraint['constraint_type'] == 'FOREIGN KEY':
                constraint_info += f" -> {constraint['foreign_table_name']}.{constraint['foreign_column_name']}"
            print(constraint_info)
        
        cursor.close()
        conn.close()
        
        # Summary
        print(f"\n=== SUMMARY ===")
        print(f"Expected OCR tables: {len(expected_tables)}")
        print(f"Actually found: {len(found_ocr_tables)}")
        
        if len(found_ocr_tables) == len(expected_tables):
            print("✅ All expected tables are present!")
        else:
            missing = set(expected_tables) - set(found_ocr_tables)
            print(f"❌ Missing tables: {missing}")
        
        return len(found_ocr_tables) == len(expected_tables)
        
    except Exception as e:
        print(f"✗ Database check failed: {e}")
        return False

if __name__ == "__main__":
    success = check_all_tables()
    sys.exit(0 if success else 1)