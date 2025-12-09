import os
import base64
import mimetypes
import logging
import uuid
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple, BinaryIO


logger = logging.getLogger(__name__)


def platform_is_mac():
    return os.sys.platform == "darwin"


class FileProcessor:
    """
    Process uploaded files for multimodal LLM analysis.
    Supports images, PDFs, text files, and binary files.
    """
    
    def __init__(self, upload_dir: str = os.getenv("UPLOAD_DIR", "uploads")):
        """
        Initialize the file processor.
        
        Args:
            upload_dir: Directory where uploaded files are stored
        """
        self.upload_dir = upload_dir
        # Ensure upload directory exists
        os.makedirs(self.upload_dir, exist_ok=True)
    
    def save_uploaded_file(self, file_obj: BinaryIO, original_filename: str) -> Tuple[str, str, str]:
        """
        Save an uploaded file with a unique ID prefix.
        
        Args:
            file_obj: File object to save (e.g., from FastAPI UploadFile.file)
            original_filename: Original name of the uploaded file
            
        Returns:
            Tuple of (file_id, original_filename, saved_file_path)
            
        Raises:
            Exception: If file save operation fails
        """
        # Generate unique file ID
        file_id = str(uuid.uuid4())
        
        # Create filename with ID prefix to avoid collisions
        saved_filename = f"{file_id}_{original_filename}"
        file_path = os.path.join(self.upload_dir, saved_filename)
        
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file_obj, buffer)
        
        logger.info(f"Saved uploaded file: {saved_filename}")
        
        return file_id, original_filename, file_path
    
    def find_file_by_id(self, file_id: str) -> Optional[tuple[str, str]]:
        """
        Find a file by its UUID.
        
        Args:
            file_id: UUID of the uploaded file
            
        Returns:
            Tuple of (file_path, file_name) if found, None otherwise
        """
        if not os.path.exists(self.upload_dir):
            return None
        
        for filename in os.listdir(self.upload_dir):
            if filename.startswith(file_id + "_"):
                file_path = os.path.join(self.upload_dir, filename)
                file_name = filename[len(file_id)+1:]  # Remove ID_ prefix
                return file_path, file_name
        
        return None
    
    def process_image_file(self, file_path: str, file_name: str, mime_type: str) -> Dict:
        """
        Process an image file by encoding it to base64.
        
        Args:
            file_path: Path to the image file
            file_name: Name of the file
            mime_type: MIME type of the image
            
        Returns:
            Dict containing image metadata and base64 data
        """
        try:
            with open(file_path, "rb") as img_file:
                img_data = base64.b64encode(img_file.read()).decode('utf-8')
                logger.info(f"Loaded image file: {file_name} ({mime_type})")
                return {
                    "type": "image",
                    "name": file_name,
                    "mime_type": mime_type,
                    "data": img_data,
                    "format": "base64"
                }
        except Exception as e:
            logger.error(f"Failed to read image {file_name}: {e}")
            return None
    
    def process_pdf_file(self, file_path: str, file_name: str) -> Dict:
        """
        Process a PDF file by extracting text content.
        Falls back to base64 encoding if PyPDF2 is not available.
        
        Args:
            file_path: Path to the PDF file
            file_name: Name of the file
            
        Returns:
            Dict containing PDF content or base64 data
        """
        try:
            import PyPDF2
            with open(file_path, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                text_content = ""
                for page in pdf_reader.pages:
                    text_content += page.extract_text() + "\n"
                
                logger.info(f"Loaded PDF file: {file_name} ({len(pdf_reader.pages)} pages)")
                return {
                    "type": "document",
                    "name": file_name,
                    "mime_type": "application/pdf",
                    "content": text_content,
                    "pages": len(pdf_reader.pages)
                }
        except ImportError:
            logger.warning("PyPDF2 not installed, treating PDF as binary")
            try:
                with open(file_path, "rb") as pdf_file:
                    pdf_data = base64.b64encode(pdf_file.read()).decode('utf-8')
                    return {
                        "type": "document",
                        "name": file_name,
                        "mime_type": "application/pdf",
                        "data": pdf_data,
                        "format": "base64"
                    }
            except Exception as e:
                logger.error(f"Failed to read PDF {file_name}: {e}")
                return None
        except Exception as e:
            logger.error(f"Failed to read PDF {file_name}: {e}")
            return None
    
    def process_text_file(self, file_path: str, file_name: str, mime_type: str) -> Dict:
        """
        Process a text file by reading its content.
        
        Args:
            file_path: Path to the text file
            file_name: Name of the file
            mime_type: MIME type of the file
            
        Returns:
            Dict containing text content
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as text_file:
                text_content = text_file.read()
                logger.info(f"Loaded text file: {file_name}")
                return {
                    "type": "document",
                    "name": file_name,
                    "mime_type": mime_type or "text/plain",
                    "content": text_content
                }
        except Exception as e:
            logger.error(f"Failed to read text file {file_name}: {e}")
            return None
    
    def process_binary_file(self, file_path: str, file_name: str, mime_type: str) -> Dict:
        """
        Process a binary file by encoding it to base64.
        
        Args:
            file_path: Path to the binary file
            file_name: Name of the file
            mime_type: MIME type of the file
            
        Returns:
            Dict containing binary data in base64 format
        """
        try:
            with open(file_path, "rb") as bin_file:
                bin_data = base64.b64encode(bin_file.read()).decode('utf-8')
                logger.info(f"Loaded binary file: {file_name}")
                return {
                    "type": "binary",
                    "name": file_name,
                    "mime_type": mime_type or "application/octet-stream",
                    "data": bin_data,
                    "format": "base64"
                }
        except Exception as e:
            logger.error(f"Failed to read binary file {file_name}: {e}")
            return None
    
    def process_file(self, file_id: str) -> Optional[Dict]:
        """
        Process a single file by ID.
        
        Args:
            file_id: UUID of the uploaded file
            
        Returns:
            Dict containing processed file data, or None if processing failed
        """
        # Find file
        file_info = self.find_file_by_id(file_id)
        if not file_info:
            logger.warning(f"File not found for ID: {file_id}")
            return None
        
        file_path, file_name = file_info
        
        # Detect MIME type
        mime_type, _ = mimetypes.guess_type(file_path)
        file_ext = Path(file_path).suffix.lower()
        
        # Process based on file type
        # Images
        if mime_type and mime_type.startswith('image/'):
            return self.process_image_file(file_path, file_name, mime_type)
        
        # PDF files
        elif file_ext == '.pdf':
            return self.process_pdf_file(file_path, file_name)
        
        # Text files
        elif file_ext in ['.txt', '.md', '.json', '.csv', '.log', 
                          '.py', '.js', '.html', '.css', '.xml', '.yaml', '.yml']:
            return self.process_text_file(file_path, file_name, mime_type)
        
        # Binary/unknown files
        else:
            return self.process_binary_file(file_path, file_name, mime_type)
    
    def process_files(self, file_ids: List[str]) -> List[Dict]:
        """
        Process multiple files by their IDs.
        
        Args:
            file_ids: List of file UUIDs
            
        Returns:
            List of processed file data dicts
        """
        file_contents = []
        
        for file_id in file_ids:
            file_data = self.process_file(file_id)
            if file_data:
                file_contents.append(file_data)
        
        return file_contents
    
    def format_file_context(self, file_contents: List[Dict]) -> str:
        """
        Format processed file contents into a context string for LLM.
        
        Args:
            file_contents: List of processed file data
            
        Returns:
            Formatted string to append to user message
        """
        if not file_contents:
            return ""
        
        file_context_parts = []
        
        for fc in file_contents:
            if fc["type"] == "image":
                # For images, note that we have the image 
                # (actual multimodal processing depends on model support)
                file_context_parts.append(
                    f"\n\n[Image: {fc['name']} - {fc['mime_type']}]"
                )
                # Note: Full multimodal image processing would require a vision-capable model
                
            elif fc["type"] == "document" and "content" in fc:
                # For text documents, include the content
                file_context_parts.append(
                    f"\n\n--- Document: {fc['name']} ---\n{fc['content']}\n--- End of {fc['name']} ---"
                )
                
            elif fc["type"] == "binary":
                file_context_parts.append(
                    f"\n\n[Binary File: {fc['name']} - {fc['mime_type']}]"
                )
        
        return "\n".join(file_context_parts)
    
    def inject_file_context_to_messages(self, messages: List, file_ids: List[str]) -> List:
        """
        Process files and inject their context into the last user message.
        
        Args:
            messages: List of message objects with 'role' and 'content' attributes
            file_ids: List of file UUIDs to process
            
        Returns:
            Modified messages list with file context injected
        """
        if not file_ids:
            return messages
        
        # Process all files
        file_contents = self.process_files(file_ids)
        
        if not file_contents:
            return messages
        
        # Format file context
        file_context = self.format_file_context(file_contents)
        
        if not file_context:
            return messages
        
        # Find the last user message and append file context
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].role == "user":
                messages[i].content = messages[i].content + file_context
                break
        
        return messages