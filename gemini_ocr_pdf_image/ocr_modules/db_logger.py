"""
PostgreSQL Database Logger for OCR Processing

Handles all database logging operations including:
- Session tracking across multiple OCR runs
- Processing logs for individual files/pages
- Error logging and exception tracking
- Connection management and validation
"""

import os
import uuid
import socket
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from contextlib import contextmanager

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    from psycopg2.pool import SimpleConnectionPool
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False


class DatabaseLogger:
    """PostgreSQL database logger for OCR processing."""
    
    def __init__(self, database_url: str, enabled: bool = True, 
                 connection_timeout: int = 30, retry_attempts: int = 3):
        """
        Initialize database logger.
        
        Args:
            database_url: PostgreSQL connection URL
            enabled: Whether database logging is enabled
            connection_timeout: Connection timeout in seconds
            retry_attempts: Number of retry attempts for connections
        """
        self.database_url = database_url
        self.enabled = enabled
        self.connection_timeout = connection_timeout
        self.retry_attempts = retry_attempts
        self.hostname = socket.gethostname()
        self.current_session_id = None
        self.connection_pool = None
        
        # Setup local logger for database operations
        self.logger = logging.getLogger('DatabaseLogger')
        
        if not self.enabled:
            self.logger.info("Database logging is disabled")
            return
            
        if not PSYCOPG2_AVAILABLE:
            raise ImportError(
                "psycopg2-binary is required for database logging. "
                "Install it with: pip install psycopg2-binary"
            )
        
        # Validate connection and setup
        self._validate_connection()
        self._setup_connection_pool()
        self._initialize_schema()
        
        self.logger.info(f"Database logger initialized for host: {self.hostname}")
    
    def _validate_connection(self):
        """Validate database connection at startup."""
        if not self.enabled:
            return
            
        for attempt in range(self.retry_attempts):
            try:
                conn = psycopg2.connect(
                    self.database_url, 
                    connect_timeout=self.connection_timeout
                )
                conn.close()
                self.logger.info(f"Database connection validated successfully")
                return
            except Exception as e:
                self.logger.error(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt == self.retry_attempts - 1:
                    raise ConnectionError(
                        f"Failed to connect to database after {self.retry_attempts} attempts: {e}"
                    )
    
    def _setup_connection_pool(self):
        """Setup connection pool for database operations."""
        if not self.enabled:
            return
            
        try:
            self.connection_pool = SimpleConnectionPool(
                minconn=1,
                maxconn=5,
                dsn=self.database_url
            )
            self.logger.info("Database connection pool created")
        except Exception as e:
            self.logger.error(f"Failed to create connection pool: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        if not self.enabled or not self.connection_pool:
            yield None
            return
            
        conn = None
        try:
            conn = self.connection_pool.getconn()
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            self.logger.error(f"Database operation failed: {e}")
            raise
        finally:
            if conn:
                self.connection_pool.putconn(conn)
    
    def _initialize_schema(self):
        """Initialize database schema if tables don't exist."""
        if not self.enabled:
            return
            
        schema_sql = """
        -- OCR processing sessions
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
        CREATE INDEX IF NOT EXISTS idx_processing_logs_session_id ON processing_logs(session_id);
        CREATE INDEX IF NOT EXISTS idx_processing_logs_status ON processing_logs(status);
        CREATE INDEX IF NOT EXISTS idx_error_logs_session_id ON error_logs(session_id);
        CREATE INDEX IF NOT EXISTS idx_error_logs_severity ON error_logs(severity);
        """
        
        with self.get_connection() as conn:
            if conn:
                cursor = conn.cursor()
                cursor.execute(schema_sql)
                conn.commit()
                cursor.close()
                self.logger.info("Database schema initialized successfully")
    
    def start_session(self, input_path: str, input_type: str, output_path: str, 
                     configuration: Dict[str, Any]) -> str:
        """
        Start a new OCR processing session.
        
        Args:
            input_path: Path to input file/directory
            input_type: Type of input ('pdf', 'image', 'directory')
            output_path: Output directory path
            configuration: OCR configuration parameters
            
        Returns:
            session_id: Unique session identifier
        """
        if not self.enabled:
            return str(uuid.uuid4())
            
        session_id = str(uuid.uuid4())
        self.current_session_id = session_id
        
        with self.get_connection() as conn:
            if conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO ocr_sessions 
                    (session_id, hostname, input_path, input_type, output_path, configuration)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (session_id, self.hostname, input_path, input_type, output_path, 
                     json.dumps(configuration)))
                conn.commit()
                cursor.close()
        
        self.logger.info(f"Started session {session_id} for {input_type}: {input_path}")
        return session_id
    
    def update_session(self, session_id: str, **kwargs):
        """Update session with new information."""
        if not self.enabled:
            return
            
        if not kwargs:
            return
            
        # Build dynamic update query
        set_clauses = []
        values = []
        
        for key, value in kwargs.items():
            if key in ['status', 'total_files', 'completed_files', 'failed_files', 'end_time']:
                set_clauses.append(f"{key} = %s")
                values.append(value)
        
        if not set_clauses:
            return
            
        values.append(session_id)
        query = f"UPDATE ocr_sessions SET {', '.join(set_clauses)} WHERE session_id = %s"
        
        with self.get_connection() as conn:
            if conn:
                cursor = conn.cursor()
                cursor.execute(query, values)
                conn.commit()
                cursor.close()
    
    def end_session(self, session_id: str, status: str = 'completed'):
        """End an OCR processing session."""
        if not self.enabled:
            return
            
        with self.get_connection() as conn:
            if conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE ocr_sessions 
                    SET status = %s, end_time = NOW()
                    WHERE session_id = %s
                """, (status, session_id))
                conn.commit()
                cursor.close()
        
        self.logger.info(f"Ended session {session_id} with status: {status}")
    
    def log_processing_start(self, session_id: str, file_path: str, page_number: Optional[int] = None):
        """Log the start of file/page processing."""
        if not self.enabled:
            return
            
        with self.get_connection() as conn:
            if conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO processing_logs 
                    (session_id, file_path, page_number, processing_start)
                    VALUES (%s, %s, %s, NOW())
                    RETURNING id
                """, (session_id, file_path, page_number))
                result = cursor.fetchone()
                conn.commit()
                cursor.close()
                return result[0] if result else None
    
    def log_processing_complete(self, session_id: str, file_path: str, page_number: Optional[int],
                              status: str, legibility_score: Optional[float] = None,
                              semantic_score: Optional[float] = None, ocr_confidence: Optional[float] = None,
                              processing_time: Optional[float] = None, text_clarity: Optional[str] = None,
                              image_quality: Optional[str] = None, ocr_prediction: Optional[str] = None,
                              semantic_prediction: Optional[str] = None, visible_text_sample: Optional[str] = None,
                              language_detected: Optional[str] = None, issues_found: Optional[str] = None,
                              error_message: Optional[str] = None):
        """Log completion of file/page processing with all assessment details."""
        if not self.enabled:
            return
            
        with self.get_connection() as conn:
            if conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO processing_logs 
                    (session_id, file_path, page_number, processing_end, status, 
                     legibility_score, semantic_score, ocr_confidence, processing_time,
                     text_clarity, image_quality, ocr_prediction, semantic_prediction,
                     visible_text_sample, language_detected, issues_found, error_message)
                    VALUES (%s, %s, %s, NOW(), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
                """, (session_id, file_path, page_number, status, legibility_score, 
                     semantic_score, ocr_confidence, processing_time, text_clarity,
                     image_quality, ocr_prediction, semantic_prediction, visible_text_sample,
                     language_detected, issues_found, error_message))
                conn.commit()
                cursor.close()
    
    def log_error(self, error_type: str, error_message: str, stack_trace: Optional[str] = None,
                  file_path: Optional[str] = None, function_name: Optional[str] = None,
                  line_number: Optional[int] = None, severity: str = 'medium',
                  session_id: Optional[str] = None):
        """Log system errors and exceptions."""
        if not self.enabled:
            return
            
        if not session_id:
            session_id = self.current_session_id
            
        with self.get_connection() as conn:
            if conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO error_logs 
                    (session_id, error_type, error_message, stack_trace, file_path,
                     function_name, line_number, severity, hostname)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (session_id, error_type, error_message, stack_trace, file_path,
                     function_name, line_number, severity, self.hostname))
                conn.commit()
                cursor.close()
        
        self.logger.error(f"Logged {severity} error: {error_type} - {error_message}")
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a processing session."""
        if not self.enabled:
            return {}
            
        with self.get_connection() as conn:
            if conn:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                
                # Get session info
                cursor.execute("""
                    SELECT * FROM ocr_sessions WHERE session_id = %s
                """, (session_id,))
                session = cursor.fetchone()
                
                # Get processing stats
                cursor.execute("""
                    SELECT 
                        status,
                        COUNT(*) as count,
                        AVG(processing_time) as avg_processing_time,
                        AVG(legibility_score) as avg_legibility_score,
                        AVG(semantic_score) as avg_semantic_score
                    FROM processing_logs 
                    WHERE session_id = %s 
                    GROUP BY status
                """, (session_id,))
                processing_stats = cursor.fetchall()
                
                # Get error count
                cursor.execute("""
                    SELECT severity, COUNT(*) as count
                    FROM error_logs 
                    WHERE session_id = %s 
                    GROUP BY severity
                """, (session_id,))
                error_stats = cursor.fetchall()
                
                cursor.close()
                
                return {
                    'session': dict(session) if session else {},
                    'processing_stats': [dict(row) for row in processing_stats],
                    'error_stats': [dict(row) for row in error_stats]
                }
        
        return {}
    
    def close(self):
        """Close database connections."""
        if self.connection_pool:
            self.connection_pool.closeall()
            self.logger.info("Database connections closed")