"""Document processor using Docling for multi-format document processing"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
import logging

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Process documents using Docling for extraction"""
    
    def __init__(self):
        """Initialize the document converter"""
        self.converter = DocumentConverter()
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a single document file
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary containing extracted text, metadata, and structure
        """
        file_path_obj = Path(file_path)
        
        if not file_path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Processing document: {file_path}")
        
        # Determine file type
        file_extension = file_path_obj.suffix.lower()
        file_type = self._get_file_type(file_extension)
        
        try:
            # Convert document using Docling
            result = self.converter.convert(file_path)
            
            # Extract text content
            text_content = result.document.export_to_markdown() if hasattr(result.document, 'export_to_markdown') else str(result.document)
            
            # Extract metadata
            metadata = {
                "file_name": file_path_obj.name,
                "file_path": str(file_path_obj),
                "file_type": file_type,
                "file_size": file_path_obj.stat().st_size,
                "pages": getattr(result.document, 'page_count', None),
            }
            
            # Extract structure if available
            structure = {}
            if hasattr(result.document, 'tables'):
                structure["tables"] = len(result.document.tables)
            if hasattr(result.document, 'images'):
                structure["images"] = len(result.document.images)
            
            return {
                "text": text_content,
                "metadata": metadata,
                "structure": structure
            }
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            raise
    
    def process_directory(self, directory_path: str, recursive: bool = True) -> List[Dict[str, Any]]:
        """
        Process all documents in a directory
        
        Args:
            directory_path: Path to directory containing documents
            recursive: Whether to process subdirectories recursively
            
        Returns:
            List of processed document dictionaries
        """
        directory = Path(directory_path)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        processed_docs = []
        supported_extensions = {'.pdf', '.docx', '.doc', '.txt', '.md', '.markdown'}
        
        pattern = "**/*" if recursive else "*"
        for file_path in directory.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                try:
                    doc_data = self.process_file(str(file_path))
                    processed_docs.append(doc_data)
                except Exception as e:
                    logger.warning(f"Failed to process {file_path}: {str(e)}")
                    continue
        
        logger.info(f"Processed {len(processed_docs)} documents from {directory_path}")
        return processed_docs
    
    def _get_file_type(self, extension: str) -> str:
        """Map file extension to file type"""
        extension_map = {
            '.pdf': 'pdf',
            '.docx': 'docx',
            '.doc': 'doc',
            '.txt': 'text',
            '.md': 'markdown',
            '.markdown': 'markdown',
        }
        return extension_map.get(extension, 'unknown')

