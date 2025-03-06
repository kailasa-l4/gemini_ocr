import os
import sys
import json
import time
import sqlite3
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.live import Live
from rich.syntax import Syntax
from rich.prompt import Prompt
import keyboard

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("chunk_viewer.log")]
)
logger = logging.getLogger("windows_chunk_viewer")

class WindowsChunkViewer:
    """Windows-compatible chunk viewer with navigation controls using rich."""
    
    def __init__(self, db_path: str = "book_processing.db"):
        """Initialize the chunk viewer."""
        self.db_path = db_path
        self.console = Console()
        self.layout = self._create_layout()
        self.current_book_id = None
        self.last_checked = datetime.now()
        
        # Viewing state
        self.display_mode = "latest"  # 'latest' or 'navigate'
        self.current_chunk_idx = 0
        self.original_scroll_lines = 30  # Show this many lines at a time
        self.processed_scroll_lines = 30
        self.original_scroll_offset = 0
        self.processed_scroll_offset = 0
        
        # Processing data
        self.processing_data = {
            "book_name": "None",
            "current_chunk": 0,
            "total_chunks": 0,
            "start_time": None,
            "original_text": "",
            "processed_text": "",
            "status": "waiting",
            "all_chunks": []  # To store all chunks for navigation
        }
        
        # Create additional table for tracking chunk processing if it doesn't exist
        self._init_database()
    
    def _init_database(self):
        """Initialize or update database schema for chunk tracking."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Add chunk_processing table if not exists
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS chunk_processing (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            book_id TEXT,
            chunk_number INTEGER,
            original_text TEXT,
            processed_text TEXT,
            status TEXT DEFAULT 'pending',
            started_at TEXT,
            completed_at TEXT,
            FOREIGN KEY (book_id) REFERENCES books(book_id)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def _create_layout(self) -> Layout:
        """Create the layout for the terminal UI."""
        layout = Layout(name="root")
        
        # Split into info panel and content panels
        layout.split(
            Layout(name="header", size=7),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=2)
        )
        
        # Split main area into original and processed
        layout["main"].split_row(
            Layout(name="original", ratio=1),
            Layout(name="processed", ratio=1)
        )
        
        return layout
    
    def _update_layout(self):
        """Update the layout with current processing data."""
        # Create header
        book_name = self.processing_data["book_name"]
        chunk_num = self.processing_data["current_chunk"]
        total_chunks = self.processing_data["total_chunks"]
        status = self.processing_data["status"]
        
        # Format timestamps
        start_time = self.processing_data["start_time"]
        start_str = start_time.strftime("%H:%M:%S") if start_time else "N/A"
        
        # Create elapsed time
        elapsed = "N/A"
        if start_time:
            elapsed_seconds = (datetime.now() - start_time).total_seconds()
            elapsed = f"{int(elapsed_seconds // 60)}m {int(elapsed_seconds % 60)}s"
        
        # Create header table
        header_table = Table(show_header=False, expand=True)
        header_table.add_column("Property")
        header_table.add_column("Value")
        
        header_table.add_row("Book", book_name)
        header_table.add_row("Mode", f"[bold cyan]{self.display_mode.upper()}[/bold cyan]")
        
        progress_text = f"Chunk {chunk_num}/{total_chunks}" if total_chunks > 0 else "No chunks"
        if self.display_mode == "navigate" and total_chunks > 0:
            progress_text += f" (Viewing chunk {self.current_chunk_idx + 1}/{len(self.processing_data['all_chunks'])})"
        header_table.add_row("Progress", progress_text)
        
        status_style = "green" if status == "completed" else "yellow"
        if status == "error":
            status_style = "red"
        header_table.add_row("Status", f"[bold {status_style}]{status.capitalize()}[/bold {status_style}]")
        header_table.add_row("Started", f"{start_str} (elapsed: {elapsed})")
        
        # Create navigation info
        nav_info = Text.from_markup(
            "\n[bold]CONTROLS:[/bold] [cyan]N[/cyan]:Next Chunk | [cyan]P[/cyan]:Previous Chunk | "
            "[cyan]M[/cyan]:Toggle Mode | [cyan]O↑/↓[/cyan]:Scroll Original | [cyan]R↑/↓[/cyan]:Scroll Processed | "
            "[cyan]Q[/cyan]:Quit"
        )
        
        header_panel = Panel(
            Table.grid(expand=True).add_row(header_table).add_row(nav_info),
            title="Chunk Processing Monitor",
            border_style="blue"
        )
        
        # Create original text panel with scrolling support
        original_text = self.processing_data["original_text"]
        
        if original_text:
            original_lines = original_text.split("\n")
            # Apply scrolling
            total_original_lines = len(original_lines)
            max_original_offset = max(0, total_original_lines - self.original_scroll_lines)
            self.original_scroll_offset = min(self.original_scroll_offset, max_original_offset)
            
            visible_original_lines = original_lines[self.original_scroll_offset:
                                                    self.original_scroll_offset + self.original_scroll_lines]
            visible_original_text = "\n".join(visible_original_lines)
            
            # Add scroll indicators
            scroll_info = ""
            if self.original_scroll_offset > 0:
                scroll_info += "↑ "
            if self.original_scroll_offset < max_original_offset:
                scroll_info += "↓"
            if scroll_info:
                scroll_info = f" [Scroll: {scroll_info}]"
            
            # Calculate lines skipped
            skipped_info = ""
            if self.original_scroll_offset > 0:
                skipped_info = f" [Lines skipped: {self.original_scroll_offset}]"
            
            original_syntax = Syntax(
                visible_original_text, 
                "markdown", 
                theme="monokai",
                line_numbers=True,
                start_line=self.original_scroll_offset + 1,
                word_wrap=True
            )
            original_panel = Panel(
                original_syntax, 
                title=f"Original Text{scroll_info}{skipped_info}",
                border_style="cyan"
            )
        else:
            original_panel = Panel("No chunk being processed", title="Original Text", border_style="cyan")
        
        # Create processed text panel with scrolling support
        processed_text = self.processing_data["processed_text"]
        
        if processed_text:
            processed_lines = processed_text.split("\n")
            # Apply scrolling
            total_processed_lines = len(processed_lines)
            max_processed_offset = max(0, total_processed_lines - self.processed_scroll_lines)
            self.processed_scroll_offset = min(self.processed_scroll_offset, max_processed_offset)
            
            visible_processed_lines = processed_lines[self.processed_scroll_offset:
                                                     self.processed_scroll_offset + self.processed_scroll_lines]
            visible_processed_text = "\n".join(visible_processed_lines)
            
            # Add scroll indicators
            scroll_info = ""
            if self.processed_scroll_offset > 0:
                scroll_info += "↑ "
            if self.processed_scroll_offset < max_processed_offset:
                scroll_info += "↓"
            if scroll_info:
                scroll_info = f" [Scroll: {scroll_info}]"
            
            # Calculate lines skipped
            skipped_info = ""
            if self.processed_scroll_offset > 0:
                skipped_info = f" [Lines skipped: {self.processed_scroll_offset}]"
            
            processed_syntax = Syntax(
                visible_processed_text, 
                "markdown", 
                theme="monokai",
                line_numbers=True,
                start_line=self.processed_scroll_offset + 1,
                word_wrap=True
            )
            processed_panel = Panel(
                processed_syntax, 
                title=f"Processed Text{scroll_info}{skipped_info}",
                border_style="green"
            )
        else:
            processed_panel = Panel(
                "Not processed yet" if status == "processing" else "No text available", 
                title="Processed Text", 
                border_style="green"
            )
        
        # Create footer
        last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        footer = Panel(
            f"Last updated: {last_updated}", 
            border_style="blue"
        )
        
        # Update layout
        self.layout["header"].update(header_panel)
        self.layout["original"].update(original_panel)
        self.layout["processed"].update(processed_panel)
        self.layout["footer"].update(footer)
        
        return self.layout
    
    def check_for_updates(self) -> bool:
        """Check for updates in the database and update processing_data."""
        now = datetime.now()
        if (now - self.last_checked).total_seconds() < 1:
            return False
        
        self.last_checked = now
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # First, check for any active book processing
        cursor.execute('''
        SELECT book_id, book_name FROM books
        WHERE status = 'processing'
        LIMIT 1
        ''')
        
        book_result = cursor.fetchone()
        if not book_result:
            # No active processing, clear data
            if self.processing_data["status"] != "waiting":
                self.processing_data = {
                    "book_name": "None",
                    "current_chunk": 0,
                    "total_chunks": 0,
                    "start_time": None,
                    "original_text": "",
                    "processed_text": "",
                    "status": "waiting",
                    "all_chunks": []
                }
                conn.close()
                return True
            return False
        
        book_id, book_name = book_result
        
        # If book changed, reset chunk info
        if book_id != self.current_book_id:
            self.current_book_id = book_id
            self.current_chunk_idx = 0
            self.original_scroll_offset = 0
            self.processed_scroll_offset = 0
            self.processing_data = {
                "book_name": book_name,
                "current_chunk": 0,
                "total_chunks": 0,
                "start_time": datetime.now(),
                "original_text": "",
                "processed_text": "",
                "status": "processing",
                "all_chunks": []
            }
        
        # Get all chunks for this book
        cursor.execute('''
        SELECT chunk_number, original_text, processed_text, status, started_at, completed_at
        FROM chunk_processing
        WHERE book_id = ?
        ORDER BY chunk_number
        ''', (book_id,))
        
        chunks = cursor.fetchall()
        
        # Store all chunks for navigation
        all_chunks = []
        for chunk in chunks:
            chunk_num, original_text, processed_text, status, started_at, completed_at = chunk
            all_chunks.append({
                "chunk_number": chunk_num,
                "original_text": original_text or "",
                "processed_text": processed_text or "",
                "status": status,
                "started_at": started_at,
                "completed_at": completed_at
            })
        
        self.processing_data["all_chunks"] = all_chunks
        self.processing_data["total_chunks"] = len(all_chunks)
        
        # Update current chunk based on mode
        if self.display_mode == "latest":
            # Get the most recent active chunk
            cursor.execute('''
            SELECT chunk_number, original_text, processed_text, status, started_at
            FROM chunk_processing
            WHERE book_id = ? AND status IN ('processing', 'completed')
            ORDER BY chunk_number DESC
            LIMIT 1
            ''', (book_id,))
            
            latest_chunk = cursor.fetchone()
            
            if latest_chunk:
                chunk_num, original_text, processed_text, status, started_at = latest_chunk
                
                # Update processing data
                self.processing_data.update({
                    "current_chunk": chunk_num,
                    "original_text": original_text or "",
                    "processed_text": processed_text or "",
                    "status": status
                })
                
                # Update current chunk index for navigation
                for i, chunk in enumerate(all_chunks):
                    if chunk["chunk_number"] == chunk_num:
                        self.current_chunk_idx = i
                        break
                
                if started_at and not self.processing_data["start_time"]:
                    try:
                        self.processing_data["start_time"] = datetime.fromisoformat(started_at)
                    except ValueError:
                        self.processing_data["start_time"] = datetime.now()
        else:  # Navigate mode
            # Ensure valid index
            if self.processing_data["all_chunks"]:
                if self.current_chunk_idx >= len(self.processing_data["all_chunks"]):
                    self.current_chunk_idx = len(self.processing_data["all_chunks"]) - 1
                
                selected_chunk = self.processing_data["all_chunks"][self.current_chunk_idx]
                self.processing_data.update({
                    "current_chunk": selected_chunk["chunk_number"],
                    "original_text": selected_chunk["original_text"],
                    "processed_text": selected_chunk["processed_text"],
                    "status": selected_chunk["status"]
                })
        
        conn.close()
        return True
    
    def handle_keypress(self, key_event):
        """Handle key presses for navigation."""
        if not hasattr(key_event, 'name'):
            return
        
        key = key_event.name
        
        # Quit on q
        if key == 'q':
            self.running = False
            return
        
        # Mode toggle
        if key == 'm':
            self.display_mode = "navigate" if self.display_mode == "latest" else "latest"
            self.original_scroll_offset = 0
            self.processed_scroll_offset = 0
            return
        
        # Next chunk in navigation mode
        if key == 'n' and self.display_mode == "navigate":
            if self.processing_data["all_chunks"]:
                self.current_chunk_idx = min(self.current_chunk_idx + 1, len(self.processing_data["all_chunks"]) - 1)
                self.original_scroll_offset = 0
                self.processed_scroll_offset = 0
            return
        
        # Previous chunk in navigation mode
        if key == 'p' and self.display_mode == "navigate":
            if self.processing_data["all_chunks"]:
                self.current_chunk_idx = max(0, self.current_chunk_idx - 1)
                self.original_scroll_offset = 0
                self.processed_scroll_offset = 0
            return
        
        # Original text scrolling
        if key == 'o+up':
            self.original_scroll_offset = max(0, self.original_scroll_offset - 5)
            return
        
        if key == 'o+down':
            self.original_scroll_offset += 5
            # Upper bound check is done during rendering
            return
        
        # Processed text scrolling
        if key == 'r+up':
            self.processed_scroll_offset = max(0, self.processed_scroll_offset - 5)
            return
        
        if key == 'r+down':
            self.processed_scroll_offset += 5
            # Upper bound check is done during rendering
            return
    
    def display(self, refresh_interval: float = 0.5):
        """Display the live terminal UI."""
        self.running = True
        self.console.clear()
        
        # Setup keyboard hooks
        keyboard.on_press(self.handle_keypress)
        
        # Register combination keys
        keyboard.add_hotkey('o+up', lambda: self.handle_keypress(type('obj', (object,), {'name': 'o+up'})))
        keyboard.add_hotkey('o+down', lambda: self.handle_keypress(type('obj', (object,), {'name': 'o+down'})))
        keyboard.add_hotkey('r+up', lambda: self.handle_keypress(type('obj', (object,), {'name': 'r+up'})))
        keyboard.add_hotkey('r+down', lambda: self.handle_keypress(type('obj', (object,), {'name': 'r+down'})))
        
        try:
            with Live(self._update_layout(), refresh_per_second=1/refresh_interval, screen=True) as live:
                while self.running:
                    if self.check_for_updates():
                        live.update(self._update_layout())
                    time.sleep(refresh_interval)
        except KeyboardInterrupt:
            self.running = False
        finally:
            # Clean up keyboard hooks
            keyboard.unhook_all()
            self.console.print("\n[bold cyan]Chunk viewer closed.[/bold cyan]")


def create_test_data(db_path):
    """Create test data in the database for testing the viewer."""
    import sqlite3
    from datetime import datetime, timedelta
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Make sure tables exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS books (
            book_id TEXT PRIMARY KEY,
            book_name TEXT,
            folder_path TEXT,
            status TEXT DEFAULT 'pending',
            file_count INTEGER DEFAULT 0,
            merged_path TEXT,
            processed_path TEXT,
            metadata_path TEXT,
            error TEXT,
            started_at TEXT,
            completed_at TEXT
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS chunk_processing (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            book_id TEXT,
            chunk_number INTEGER,
            original_text TEXT,
            processed_text TEXT,
            status TEXT DEFAULT 'pending',
            started_at TEXT,
            completed_at TEXT,
            FOREIGN KEY (book_id) REFERENCES books(book_id)
        )
        ''')
        
        # Create a test book
        book_id = "test_book_id"
        
        cursor.execute('''
        INSERT OR REPLACE INTO books 
        (book_id, book_name, folder_path, status, file_count, started_at) 
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            book_id,
            "Test Book",
            "/path/to/book",
            "processing",
            5,
            datetime.now().isoformat()
        ))
        
        # Create test chunks
        now = datetime.now()
        chunks = [
            {
                "chunk_number": 1,
                "original_text": "This is the original text for chunk 1.\nIt spans multiple lines.\nLet's see how the scrolling works.",
                "processed_text": "# This is the processed text for chunk 1\n\nIt spans multiple lines.\n\nLet's see how the scrolling works.",
                "status": "completed",
                "started_at": (now - timedelta(minutes=5)).isoformat(),
                "completed_at": (now - timedelta(minutes=4)).isoformat(),
            },
            {
                "chunk_number": 2,
                "original_text": "Original text for chunk 2.\nMore lines here.\nAnd even more text that goes beyond the visible area.\n" + "Long line " * 20,
                "processed_text": "# Processed text for chunk 2\n\nMore lines here.\n\nAnd even more text that goes beyond the visible area.\n\n" + "Long line " * 20,
                "status": "completed",
                "started_at": (now - timedelta(minutes=3)).isoformat(),
                "completed_at": (now - timedelta(minutes=2)).isoformat(),
            },
            {
                "chunk_number": 3,
                "original_text": "Original text for chunk 3, which is currently being processed.\n" + "\n".join([f"Line {i} of chunk 3" for i in range(1, 30)]),
                "processed_text": "",
                "status": "processing",
                "started_at": (now - timedelta(minutes=1)).isoformat(),
                "completed_at": None,
            }
        ]
        
        for chunk in chunks:
            cursor.execute('''
            INSERT OR REPLACE INTO chunk_processing
            (book_id, chunk_number, original_text, processed_text, status, started_at, completed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                book_id,
                chunk["chunk_number"],
                chunk["original_text"],
                chunk["processed_text"],
                chunk["status"],
                chunk["started_at"],
                chunk["completed_at"]
            ))
        
        conn.commit()
        conn.close()
        print("Test data created successfully!")
        return True
    
    except Exception as e:
        print(f"Error creating test data: {str(e)}")
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Windows-Compatible Chunk Viewer")
    parser.add_argument("--db-path", default="book_processing.db", help="Path to the database file")
    parser.add_argument("--refresh", type=float, default=0.5, help="Refresh interval in seconds")
    parser.add_argument("--create-test-data", action="store_true", help="Create test data for demo")
    
    args = parser.parse_args()
    
    if args.create_test_data:
        create_test_data(args.db_path)
        return
    
    viewer = WindowsChunkViewer(args.db_path)
    viewer.display(refresh_interval=args.refresh)


if __name__ == "__main__":
    main()