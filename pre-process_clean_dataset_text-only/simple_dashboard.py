import os
import sys
import json
import time
import sqlite3
import datetime
from typing import Dict, Any, List

def get_processing_stats(db_path: str) -> Dict[str, Any]:
    """Get book processing statistics."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    stats = {
        "total": 0,
        "pending": 0,
        "merging": 0,
        "merged": 0,
        "processing": 0,
        "completed": 0,
        "error": 0
    }
    
    # Count books by status
    cursor.execute("SELECT status, COUNT(*) FROM books GROUP BY status")
    for status, count in cursor.fetchall():
        stats[status] = count
        stats["total"] += count
    
    # Get API usage stats
    cursor.execute('''
    SELECT operation, COUNT(*), SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END)
    FROM api_usage GROUP BY operation
    ''')
    
    api_stats = {}
    for operation, count, successes in cursor.fetchall():
        api_stats[operation] = {
            "total": count,
            "successes": successes,
            "failures": count - successes,
            "success_rate": round((successes / count) * 100, 2) if count > 0 else 0
        }
    
    stats["api_usage"] = api_stats
    conn.close()
    
    return stats

def get_recent_activities(db_path: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Get recent processing activities."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get recent status changes
    cursor.execute('''
    SELECT b.book_name, b.status, 
           COALESCE(b.completed_at, b.started_at) as activity_time
    FROM books b
    WHERE b.status != 'pending'
    ORDER BY activity_time DESC
    LIMIT ?
    ''', (limit,))
    
    activities = []
    for book_name, status, activity_time in cursor.fetchall():
        activities.append({
            "book_name": book_name,
            "status": status,
            "time": activity_time
        })
    
    conn.close()
    return activities

def display_simple_dashboard(db_path: str, refresh_interval: int = 5):
    """Display a simple text-based dashboard that works in any terminal."""
    try:
        while True:
            # Clear screen (works on Windows and Unix)
            os.system('cls' if os.name == 'nt' else 'clear')
            
            # Get stats
            stats = get_processing_stats(db_path)
            activities = get_recent_activities(db_path)
            last_updated = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Display header
            print("=" * 80)
            print(f"BOOK PROCESSING DASHBOARD - Last Updated: {last_updated}")
            print("=" * 80)
            
            # Display status counts
            print("\nPROCESSING STATUS:")
            print("-" * 50)
            total = stats["total"]
            for status in ["pending", "merging", "merged", "processing", "completed", "error"]:
                count = stats.get(status, 0)
                percentage = (count / total * 100) if total > 0 else 0
                print(f"{status.capitalize():<15}: {count:>5} ({percentage:>5.1f}%)")
            
            # Display recent activities
            print("\nRECENT ACTIVITIES:")
            print("-" * 50)
            if activities:
                for activity in activities:
                    print(f"{activity['book_name']:<30} - {activity['status']:<15} - {activity['time']}")
            else:
                print("No recent activities")
            
            # Display API usage
            print("\nAPI USAGE STATISTICS:")
            print("-" * 50)
            api_stats = stats.get("api_usage", {})
            if api_stats:
                for operation, data in api_stats.items():
                    print(f"{operation.replace('_', ' ').title():<20}: {data['total']:>5} calls, {data['success_rate']:>5.1f}% success")
            else:
                print("No API usage recorded")
            
            # Display progress
            total_processed = stats.get("completed", 0) + stats.get("error", 0)
            progress = (total_processed / total * 100) if total > 0 else 0
            print("\nOVERALL PROGRESS:")
            print("-" * 50)
            print(f"Processed: {total_processed}/{total} ({progress:.1f}%)")
            
            # Progress bar
            bar_width = 50
            filled_width = int(progress / 100 * bar_width)
            bar = "█" * filled_width + "░" * (bar_width - filled_width)
            print(f"[{bar}]")
            
            print("\nPress Ctrl+C to exit dashboard")
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\nDashboard closed.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple book processing dashboard")
    parser.add_argument("--db-path", default="book_processing.db", help="Path to the database file")
    parser.add_argument("--refresh", type=int, default=5, help="Refresh interval in seconds")
    
    args = parser.parse_args()
    
    display_simple_dashboard(args.db_path, args.refresh)