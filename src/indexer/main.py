"""Document indexer console application"""

import argparse
import logging
import sys
from pathlib import Path
from src.indexer.document_processor import DocumentProcessor
from src.indexer.chunking_strategy import ChunkingStrategy
from src.indexer.vector_store import VectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def index_documents(input_path: str, recursive: bool = True) -> None:
    """
    Index documents from a file or directory
    
    Args:
        input_path: Path to file or directory
        recursive: Whether to process subdirectories recursively
    """
    try:
        logger.info(f"Starting document indexing: {input_path}")
        
        # Initialize components
        processor = DocumentProcessor()
        chunker = ChunkingStrategy()
        vector_store = VectorStore()
        
        # Process documents
        input_path_obj = Path(input_path)
        
        if input_path_obj.is_file():
            # Process single file
            logger.info(f"Processing single file: {input_path}")
            doc_data = processor.process_file(input_path)
            documents = [doc_data]
        elif input_path_obj.is_dir():
            # Process directory
            logger.info(f"Processing directory: {input_path}")
            documents = processor.process_directory(input_path, recursive=recursive)
        else:
            raise ValueError(f"Invalid input path: {input_path}")
        
        if not documents:
            logger.warning("No documents found to process")
            return
        
        # Chunk documents
        logger.info("Chunking documents...")
        chunks = chunker.chunk_documents(documents)
        
        if not chunks:
            logger.warning("No chunks created from documents")
            return
        
        # Store in vector database
        logger.info("Storing documents in vector database...")
        vector_store.add_documents(chunks)
        
        logger.info(f"Successfully indexed {len(documents)} documents with {len(chunks)} chunks")
        
    except Exception as e:
        logger.error(f"Error during indexing: {str(e)}", exc_info=True)
        sys.exit(1)


def main():
    """Main entry point for the indexer"""
    parser = argparse.ArgumentParser(
        description="Document Indexer for Agentic RAG System"
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to document file or directory"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Process subdirectories recursively (default: True)"
    )
    parser.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="Do not process subdirectories recursively"
    )
    
    args = parser.parse_args()
    
    index_documents(args.input_path, recursive=args.recursive)


if __name__ == "__main__":
    main()

