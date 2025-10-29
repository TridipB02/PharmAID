"""
Internal Knowledge Agent - Document Intelligence Specialist
Extracts insights from internal company documents and reports
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import INTERNAL_DOCS_DIR


class InternalKnowledgeAgent:
    """
    Internal Knowledge Agent

    Responsibilities:
    - Search and analyze internal documents
    - Extract key insights from strategy documents
    - Summarize field reports and meeting notes
    - Retrieve historical project information
    - Provide context from company knowledge base
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize Internal Knowledge Agent

        Args:
            verbose: Whether to print detailed logs
        """
        self.verbose = verbose
        self.docs_directory = INTERNAL_DOCS_DIR

        # Ensure directory exists
        self.docs_directory.mkdir(exist_ok=True, parents=True)

        if self.verbose:
            print("✓ Internal Knowledge Agent initialized")
            print(f"  Documents directory: {self.docs_directory}")

    def search_internal(
        self, keywords: List[str] = None, document_type: str = None
    ) -> Dict[str, Any]:
        """
        Search internal documents

        Args:
            keywords: Search keywords
            document_type: Type of document to search (strategy, report, minutes, etc.)

        Returns:
            Dictionary with search results
        """
        if self.verbose:
            print(f"\n[Internal Knowledge Agent] Searching internal documents...")
            print(f"  Keywords: {keywords}")
            print(f"  Document type: {document_type}")

        results = {
            "summary": "",
            "total_documents_found": 0,
            "documents": [],
            "key_insights": [],
            "relevant_excerpts": [],
        }

        # Get list of documents in directory
        documents = self._get_available_documents()

        if not documents:
            results["summary"] = (
                "No internal documents available in the knowledge base."
            )
            if self.verbose:
                print("  ⚠ No documents found in internal directory")
            return results

        # Filter documents by keywords and type
        matching_docs = self._filter_documents(documents, keywords, document_type)

        results["total_documents_found"] = len(matching_docs)
        results["documents"] = matching_docs

        # Extract insights
        if matching_docs:
            results["key_insights"] = self._extract_insights(matching_docs, keywords)
            results["relevant_excerpts"] = self._extract_excerpts(
                matching_docs, keywords
            )

        # Generate summary
        results["summary"] = self._generate_summary(results)

        if self.verbose:
            print(
                f"✓ Search complete - {results['total_documents_found']} documents found"
            )

        return results

    def _get_available_documents(self) -> List[Dict[str, Any]]:
        """
        Get list of available documents in internal directory

        Returns:
            List of document metadata dictionaries
        """
        documents = []

        # Check for PDF, TXT, and other document files
        for file_path in self.docs_directory.iterdir():
            if file_path.is_file():
                doc_info = {
                    "filename": file_path.name,
                    "path": str(file_path),
                    "size_bytes": file_path.stat().st_size,
                    "modified": file_path.stat().st_mtime,
                    "extension": file_path.suffix.lower(),
                }
                documents.append(doc_info)

        return documents

    def _filter_documents(
        self,
        documents: List[Dict],
        keywords: List[str] = None,
        document_type: str = None,
    ) -> List[Dict[str, Any]]:
        """
        Filter documents by keywords and type

        Args:
            documents: List of document metadata
            keywords: Keywords to filter by
            document_type: Document type to filter by

        Returns:
            Filtered list of documents
        """
        filtered = documents.copy()

        # Filter by document type (based on filename)
        if document_type:
            filtered = [
                doc
                for doc in filtered
                if document_type.lower() in doc["filename"].lower()
            ]

        # Filter by keywords (basic filename matching)
        if keywords:
            keyword_filtered = []
            for doc in filtered:
                filename_lower = doc["filename"].lower()
                if any(kw.lower() in filename_lower for kw in keywords):
                    keyword_filtered.append(doc)
            filtered = keyword_filtered

        return filtered

    def _extract_insights(
        self, documents: List[Dict], keywords: List[str] = None
    ) -> List[str]:
        """
        Extract key insights from documents

        Args:
            documents: List of documents
            keywords: Search keywords

        Returns:
            List of insight strings
        """
        insights = []

        # Basic insights based on document metadata
        insights.append(f"Found {len(documents)} relevant internal document(s)")

        # Document types
        doc_types = {}
        for doc in documents:
            ext = doc["extension"]
            doc_types[ext] = doc_types.get(ext, 0) + 1

        if doc_types:
            type_summary = ", ".join(
                [f"{count} {ext}" for ext, count in doc_types.items()]
            )
            insights.append(f"Document types: {type_summary}")

        # Recent documents
        if documents:
            recent_docs = sorted(documents, key=lambda x: x["modified"], reverse=True)[
                :3
            ]
            recent_names = [doc["filename"] for doc in recent_docs]
            insights.append(f"Most recent: {', '.join(recent_names)}")

        return insights

    def _extract_excerpts(
        self, documents: List[Dict], keywords: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract relevant excerpts from documents

        Args:
            documents: List of documents
            keywords: Search keywords

        Returns:
            List of excerpt dictionaries
        """
        excerpts = []

        for doc in documents[:5]:  # Limit to first 5 documents
            # Try to read text files
            if doc["extension"] in [".txt", ".md"]:
                try:
                    with open(doc["path"], "r", encoding="utf-8") as f:
                        content = f.read()
                        # Extract first 200 characters as excerpt
                        excerpt_text = (
                            content[:200] + "..." if len(content) > 200 else content
                        )

                        excerpts.append(
                            {
                                "document": doc["filename"],
                                "excerpt": excerpt_text,
                                "relevance": (
                                    "High"
                                    if keywords
                                    and any(
                                        kw.lower() in content.lower() for kw in keywords
                                    )
                                    else "Medium"
                                ),
                            }
                        )
                except Exception as e:
                    if self.verbose:
                        print(f"  ⚠ Could not read {doc['filename']}: {e}")
            else:
                # For PDFs and other formats, just note the document
                excerpts.append(
                    {
                        "document": doc["filename"],
                        "excerpt": f"[{doc['extension'].upper()} document - content extraction requires PDF parser]",
                        "relevance": "Unknown",
                    }
                )

        return excerpts

    def _generate_summary(self, results: Dict[str, Any]) -> str:
        """
        Generate executive summary

        Args:
            results: Search results

        Returns:
            Summary string
        """
        doc_count = results["total_documents_found"]

        if doc_count == 0:
            return "No internal documents found matching the search criteria."

        summary_parts = [f"Found {doc_count} relevant internal document(s)."]

        if results["key_insights"]:
            summary_parts.append(
                f"{len(results['key_insights'])} key insight(s) extracted."
            )

        return " ".join(summary_parts)

    def get_document_summary(self, filename: str) -> Dict[str, Any]:
        """
        Get summary of a specific document

        Args:
            filename: Name of the document

        Returns:
            Document summary
        """
        if self.verbose:
            print(f"\n[Internal Knowledge Agent] Retrieving document: {filename}")

        file_path = self.docs_directory / filename

        if not file_path.exists():
            return {"filename": filename, "found": False, "error": "Document not found"}

        # Get file info
        file_stat = file_path.stat()

        summary = {
            "filename": filename,
            "found": True,
            "path": str(file_path),
            "size_bytes": file_stat.st_size,
            "size_mb": round(file_stat.st_size / (1024 * 1024), 2),
            "modified": file_stat.st_mtime,
            "extension": file_path.suffix.lower(),
        }

        # Try to read content preview
        if file_path.suffix.lower() in [".txt", ".md"]:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    summary["preview"] = (
                        content[:500] + "..." if len(content) > 500 else content
                    )
                    summary["word_count"] = len(content.split())
            except Exception as e:
                summary["preview"] = f"Could not read content: {e}"
        else:
            summary["preview"] = (
                f"[{file_path.suffix.upper()} file - binary or requires special parser]"
            )

        return summary

    def list_all_documents(self) -> Dict[str, Any]:
        """
        List all available internal documents

        Returns:
            Dictionary with document listing
        """
        if self.verbose:
            print(f"\n[Internal Knowledge Agent] Listing all documents...")

        documents = self._get_available_documents()

        # Organize by type
        by_type = {}
        for doc in documents:
            ext = doc["extension"]
            if ext not in by_type:
                by_type[ext] = []
            by_type[ext].append(doc["filename"])

        return {
            "total_documents": len(documents),
            "documents": documents,
            "by_type": by_type,
            "directory": str(self.docs_directory),
        }

    def add_document_metadata(
        self, filename: str, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Add metadata tags to a document (for future retrieval)

        Args:
            filename: Document filename
            metadata: Metadata dictionary (tags, description, etc.)

        Returns:
            Result dictionary
        """
        # This is a placeholder for metadata management
        # In production, this would store metadata in a database

        return {
            "filename": filename,
            "metadata_added": True,
            "metadata": metadata,
            "note": "Metadata storage requires database implementation",
        }


# Convenience function
def get_internal_knowledge_agent(verbose: bool = True) -> InternalKnowledgeAgent:
    """
    Get instance of Internal Knowledge Agent

    Args:
        verbose: Whether to enable verbose logging

    Returns:
        Initialized InternalKnowledgeAgent instance
    """
    return InternalKnowledgeAgent(verbose=verbose)


# Test the agent
if __name__ == "__main__":
    print("=" * 70)
    print("INTERNAL KNOWLEDGE AGENT - TEST SUITE")
    print("=" * 70)

    # Initialize agent
    agent = get_internal_knowledge_agent(verbose=True)

    # Test 1: List all documents
    print("\n" + "=" * 70)
    print("TEST 1: List All Documents")
    print("=" * 70)
    listing = agent.list_all_documents()
    print(json.dumps(listing, indent=2))

    # Test 2: Search documents
    print("\n" + "=" * 70)
    print("TEST 2: Search Internal Documents")
    print("=" * 70)
    result = agent.search_internal(keywords=["strategy", "portfolio"])
    print(json.dumps(result, indent=2))

    # Test 3: Create a sample document and search for it
    print("\n" + "=" * 70)
    print("TEST 3: Create Sample Document and Retrieve")
    print("=" * 70)

    # Create a sample document
    sample_doc_path = INTERNAL_DOCS_DIR / "sample_strategy.txt"
    with open(sample_doc_path, "w", encoding="utf-8") as f:
        f.write(
            """
Portfolio Strategy Document
Date: 2024-10-23

Key Initiatives:
1. Focus on repurposing existing molecules
2. Explore diabetes and cardiovascular therapeutic areas
3. Target emerging markets with generic formulations

Strategic Priorities:
- Metformin repurposing for oncology indications
- Atorvastatin lifecycle management
- Partnership opportunities in respiratory space

Next Steps:
- Conduct market analysis for top 10 molecules
- File patents for novel formulations
- Initiate clinical trials for repurposed indications
        """
        )

    print(f"Created sample document: {sample_doc_path.name}")

    # Search for it
    search_result = agent.search_internal(keywords=["strategy", "metformin"])
    print(f"\nSearch results: {search_result['summary']}")
    print(f"Documents found: {search_result['total_documents_found']}")

    # Get document summary
    doc_summary = agent.get_document_summary("sample_strategy.txt")
    print(f"\nDocument Summary:")
    print(json.dumps(doc_summary, indent=2))

    print("\n✓ All Internal Knowledge Agent tests completed!")
