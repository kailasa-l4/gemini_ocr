import os
import sys
import json
import time
import logging
import argparse
import sqlite3
import datetime
from typing import List, Dict, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.live import Live

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("batch_manager.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("batch_manager")

# Initialize rich console
console = Console()

class BatchManager:
    """Manage and monitor batch processing of books."""
    
    def __init__(self, db_path: str = "book_processing.db"):
        """Initialize the batch manager."""
        self.db_path = db_path
        self._check_database()
    
    def _check_database(self):
        """Check if the database exists and is valid."""
        if not os.path.exists(self.db_path):
            console.print(f"[bold red]Error:[/bold red] Database not found at {self.db_path}")
            console.print("Run the book processor script first to create and populate the database.")
            sys.exit(1)
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [table[0] for table in cursor.fetchall()]
            required_tables = ['books', 'files', 'api_usage']
            
            for table in required_tables:
                if table not in tables:
                    console.print(f"[bold red]Error:[/bold red] Required table '{table}' not found in database.")
                    sys.exit(1)
            
            conn.close()
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] Failed to connect to database: {str(e)}")
            sys.exit(1)
    
    def display_status_dashboard(self, refresh_interval: int = 5):
        """Display a live dashboard of processing status."""
        with Live(console=console, refresh_per_second=1) as live:
            while True:
                content = self._generate_dashboard()
                live.update(content)
                time.sleep(refresh_interval)
    
    def _generate_dashboard(self):
        """Generate the dashboard content."""
        stats = self.get_processing_stats()
        last_updated = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create overall stats panel
        status_table = Table(title="Processing Status")
        status_table.add_column("Status", style="cyan")
        status_table.add_column("Count", style="green")
        status_table.add_column("Percentage", style="yellow")
        
        total = stats["total"]
        for status in ["pending", "merging", "merged", "processing", "completed", "error"]:
            if total > 0:
                percentage = f"{(stats[status] / total) * 100:.1f}%"
            else:
                percentage = "0.0%"
            status_table.add_row(status.capitalize(), str(stats[status]), percentage)
        
        # Create recent activities table
        recent_table = Table(title="Recent Processing Activities")
        recent_table.add_column("Book", style="cyan")
        recent_table.add_column("Status", style="green")
        recent_table.add_column("Time", style="yellow")
        
        recent_activities = self.get_recent_activities(limit=5)
        for activity in recent_activities:
            status_style = "green" if activity["status"] == "completed" else "yellow"
            if activity["status"] == "error":
                status_style = "red"
            
            recent_table.add_row(
                activity["book_name"],
                f"[{status_style}]{activity['status'].capitalize()}[/{status_style}]",
                activity["time"]
            )
        
        # Create API usage panel
        api_table = Table(title="API Usage Statistics")
        api_table.add_column("Operation", style="cyan")
        api_table.add_column("Total", style="green")
        api_table.add_column("Success Rate", style="yellow")
        
        for operation, data in stats.get("api_usage", {}).items():
            success_style = "green" if data["success_rate"] >= 90 else "yellow"
            if data["success_rate"] < 70:
                success_style = "red"
            
            api_table.add_row(
                operation.replace("_", " ").title(),
                str(data["total"]),
                f"[{success_style}]{data['success_rate']}%[/{success_style}]"
            )
        
        # Create progress panel
        total_processed = stats["completed"] + stats["error"]
        progress_percentage = (total_processed / total) * 100 if total > 0 else 0
        
        progress_text = f"[bold]Overall Progress:[/bold] {progress_percentage:.1f}% ({total_processed}/{total})"
        progress_panel = Panel(progress_text, title="Progress", border_style="green")
        
        # Combine panels
        return Panel(
            f"{status_table}\n\n{recent_table}\n\n{api_table}\n\n{progress_panel}",
            title=f"Book Processing Dashboard [Last Updated: {last_updated}]",
            border_style="blue"
        )
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get book processing statistics."""
        conn = sqlite3.connect(self.db_path)
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
    
    def get_recent_activities(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent processing activities."""
        conn = sqlite3.connect(self.db_path)
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
    
    def retry_failed_books(self) -> int:
        """Reset failed books for retry."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Count failed books
        cursor.execute("SELECT COUNT(*) FROM books WHERE status = 'error'")
        error_count = cursor.fetchone()[0]
        
        # Reset status
        cursor.execute('''
        UPDATE books 
        SET status = 'pending', error = NULL
        WHERE status = 'error'
        ''')
        
        conn.commit()
        conn.close()
        
        console.print(f"[bold green]Reset {error_count} failed books for retry.[/bold green]")
        return error_count
    
    def prioritize_books(self, book_ids: List[str]) -> int:
        """Prioritize specific books for processing."""
        if not book_ids:
            console.print("[yellow]No book IDs provided for prioritization.[/yellow]")
            return 0
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        prioritized = 0
        for book_id in book_ids:
            # Check if book exists
            cursor.execute("SELECT book_id FROM books WHERE book_id = ?", (book_id,))
            if cursor.fetchone():
                # Add priority flag (this would require schema modification)
                cursor.execute('''
                UPDATE books 
                SET priority = 1
                WHERE book_id = ?
                ''', (book_id,))
                prioritized += 1
        
        conn.commit()
        conn.close()
        
        console.print(f"[bold green]Prioritized {prioritized} books for processing.[/bold green]")
        return prioritized
    
    def export_book_list(self, output_path: str) -> int:
        """Export list of books and their processing status."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT book_id, book_name, folder_path, status, file_count, 
               started_at, completed_at, error
        FROM books
        ORDER BY book_name
        ''')
        
        books = []
        for row in cursor.fetchall():
            book_id, book_name, folder_path, status, file_count, started_at, completed_at, error = row
            books.append({
                "book_id": book_id,
                "book_name": book_name,
                "folder_path": folder_path,
                "status": status,
                "file_count": file_count,
                "started_at": started_at,
                "completed_at": completed_at,
                "error": error
            })
        
        conn.close()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(books, f, indent=2, ensure_ascii=False)
        
        console.print(f"[bold green]Exported {len(books)} book records to {output_path}[/bold green]")
        return len(books)
    
    def show_book_details(self, book_id: str):
        """Show detailed information about a specific book."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get book details
        cursor.execute('''
        SELECT book_id, book_name, folder_path, status, file_count, 
               merged_path, processed_path, metadata_path,
               started_at, completed_at, error
        FROM books
        WHERE book_id = ?
        ''', (book_id,))
        
        book = cursor.fetchone()
        
        if not book:
            console.print(f"[bold red]Book not found with ID: {book_id}[/bold red]")
            conn.close()
            return
        
        book_id, book_name, folder_path, status, file_count, merged_path, processed_path, metadata_path, started_at, completed_at, error = book
        
        # Get file details
        cursor.execute('''
        SELECT file_id, file_path, status, error
        FROM files
        WHERE book_id = ?
        ''', (book_id,))
        
        files = cursor.fetchall()
        
        # Get API usage for this book
        cursor.execute('''
        SELECT operation, COUNT(*), SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END)
        FROM api_usage
        WHERE book_id = ?
        GROUP BY operation
        ''', (book_id,))
        
        api_usage = cursor.fetchall()
        
        conn.close()
        
        # Display book details
        console.print(Panel(f"[bold cyan]Book Details: {book_name}[/bold cyan]", border_style="blue"))
        
        table = Table(show_header=False)
        table.add_column("Property", style="cyan")
        table.add_column("Value")
        
        table.add_row("Book ID", book_id)
        table.add_row("Status", f"[bold]{status.capitalize()}[/bold]")
        table.add_row("Folder Path", folder_path)
        table.add_row("File Count", str(file_count))
        
        if merged_path:
            table.add_row("Merged File", merged_path)
        
        if processed_path:
            table.add_row("Processed File", processed_path)
        
        if metadata_path:
            table.add_row("Metadata File", metadata_path)
        
        if started_at:
            table.add_row("Started", started_at)
        
        if completed_at:
            table.add_row("Completed", completed_at)
        
        if error:
            table.add_row("Error", f"[red]{error}[/red]")
        
        console.print(table)
        
        # Display files
        if files:
            console.print("\n[bold cyan]Files:[/bold cyan]")
            file_table = Table()
            file_table.add_column("File Path", style="cyan")
            file_table.add_column("Status", style="green")
            
            for file_id, file_path, file_status, file_error in files:
                status_text = file_status.capitalize()
                if file_error:
                    status_text = f"[red]{status_text} (Error)[/red]"
                
                file_table.add_row(os.path.basename(file_path), status_text)
            
            console.print(file_table)
        
        # Display API usage
        if api_usage:
            console.print("\n[bold cyan]API Usage:[/bold cyan]")
            api_table = Table()
            api_table.add_column("Operation", style="cyan")
            api_table.add_column("Count", style="green")
            api_table.add_column("Success Rate", style="yellow")
            
            for operation, count, successes in api_usage:
                success_rate = (successes / count) * 100 if count > 0 else 0
                api_table.add_row(
                    operation.replace("_", " ").title(),
                    str(count),
                    f"{success_rate:.1f}%"
                )
            
            console.print(api_table)


def main():
    parser = argparse.ArgumentParser(description="Manage and monitor book processing")
    parser.add_argument("--db-path", default="book_processing.db", help="Path to the database file")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser("dashboard", help="Display live processing dashboard")
    dashboard_parser.add_argument("--refresh", type=int, default=5, help="Refresh interval in seconds")
    
    # Retry command
    subparsers.add_parser("retry", help="Reset failed books for retry")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export book list to JSON")
    export_parser.add_argument("--output", default="books_export.json", help="Output file path")
    
    # Book details command
    details_parser = subparsers.add_parser("details", help="Show details for a specific book")
    details_parser.add_argument("book_id", help="ID of the book to show details for")
    
    # Prioritize command
    prioritize_parser = subparsers.add_parser("prioritize", help="Prioritize specific books")
    prioritize_parser.add_argument("book_ids", nargs="+", help="IDs of books to prioritize")
    
    args = parser.parse_args()
    
    manager = BatchManager(args.db_path)
    
    if args.command == "dashboard":
        try:
            manager.display_status_dashboard(refresh_interval=args.refresh)
        except KeyboardInterrupt:
            console.print("\n[bold cyan]Dashboard closed.[/bold cyan]")
    
    elif args.command == "retry":
        manager.retry_failed_books()
    
    elif args.command == "export":
        manager.export_book_list(args.output)
    
    elif args.command == "details":
        manager.show_book_details(args.book_id)
    
    elif args.command == "prioritize":
        manager.prioritize_books(args.book_ids)
    
    else:
        # Default to showing stats if no command specified
        stats = manager.get_processing_stats()
        console.print(Panel(json.dumps(stats, indent=2), title="Processing Statistics", border_style="green"))


if __name__ == "__main__":
    main()
