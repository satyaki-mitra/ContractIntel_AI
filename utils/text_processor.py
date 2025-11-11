# DEPENDENCIES
import re
from typing import Any
from typing import List
from typing import Dict
from typing import Optional
from difflib import SequenceMatcher

# Advanced NLP (optional but recommended)
try:
    import spacy
    SPACY_AVAILABLE = True

except ImportError:
    SPACY_AVAILABLE = False
    print("[TextProcessor] spaCy not available. Install with: pip install spacy && python -m spacy download en_core_web_sm")

# Language detection
try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
    
except ImportError:
    LANGDETECT_AVAILABLE = False


class TextProcessor:
    """
    Text processing and normalization utilities
    """
    def __init__(self, use_spacy: bool = True):
        """
        Initialize text processor
        
        Arguments:
        ----------
            use_spacy { bool } : Whether to use spaCy for advanced NLP (if available)
        """
        self.nlp = None
        
        if use_spacy and SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")

                print("[TextProcessor] spaCy model loaded successfully")
            
            except OSError:
                print("[TextProcessor] spaCy model not found. Run: python -m spacy download en_core_web_sm")
                self.nlp = None
    

    @staticmethod
    def normalize_text(text: str, lowercase: bool = True, remove_special_chars: bool = False) -> str:
        """
        Normalize text for analysis
        
        Args:
            text: Input text
            lowercase: Convert to lowercase
            remove_special_chars: Remove special characters
        
        Returns:
            Normalized text
        """
        if lowercase:
            text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        if remove_special_chars:
            # Keep alphanumeric and basic punctuation
            text = re.sub(r'[^\w\s.,;:!?()\-\'\"&@#$%]', '', text)
        
        return text.strip()
    
    @staticmethod
    def split_into_paragraphs(text: str, min_length: int = 20) -> List[str]:
        """
        Split text into paragraphs
        
        Args:
            text: Input text
            min_length: Minimum paragraph length in characters
        
        Returns:
            List of paragraphs
        """
        # Split on double newlines
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Filter short and empty paragraphs
        return [p.strip() for p in paragraphs if len(p.strip()) >= min_length]
    
    @staticmethod
    def extract_sentences(text: str, min_length: int = 10) -> List[str]:
        """
        Extract sentences from text (basic method)
        
        Args:
            text: Input text
            min_length: Minimum sentence length in characters
        
        Returns:
            List of sentences
        """
        # Simple sentence splitting on .!?
        sentences = re.split(r'[.!?]+', text)
        
        # Clean and filter
        sentences = [s.strip() for s in sentences if len(s.strip()) >= min_length]
        
        return sentences
    
    def extract_sentences_advanced(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract sentences with NER and metadata using spaCy
        
        Args:
            text: Input text
        
        Returns:
            List of sentence dictionaries with entities and metadata
        """
        if not self.nlp:
            # Fallback to basic extraction
            basic_sentences = self.extract_sentences(text)
            return [{"text": s, "entities": [], "start_char": 0, "end_char": 0} 
                   for s in basic_sentences]
        
        doc = self.nlp(text[:100000])  # Limit to 100K chars for performance
        sentences = []
        
        for sent in doc.sents:
            sentences.append({
                "text": sent.text.strip(),
                "entities": [(ent.text, ent.label_) for ent in sent.ents],
                "start_char": sent.start_char,
                "end_char": sent.end_char,
                "tokens": [token.text for token in sent]
            })
        
        return sentences
    
    # =========================================================================
    # LEGAL-SPECIFIC EXTRACTION
    # =========================================================================
    
    @staticmethod
    def extract_legal_entities(text: str) -> Dict[str, List[str]]:
        """
        Extract legal-specific entities (parties, dates, amounts, references)
        
        Args:
            text: Input text
        
        Returns:
            Dictionary of extracted entities by type
        """
        entities = {
            "parties": [],
            "dates": [],
            "amounts": [],
            "addresses": [],
            "references": [],
            "emails": [],
            "phone_numbers": []
        }
        
        # Party names (PARTY A, "the Employee", Company Name Inc.)
        party_patterns = [
            r'(?:PARTY|Party)\s+[A-Z]',
            r'"the\s+\w+"',
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc|LLC|Corp|Ltd|Limited|Company)\.?',
            r'(?:the\s+)?(Employer|Employee|Consultant|Contractor|Client|Vendor|Supplier|Landlord|Tenant|Buyer|Seller)',
        ]
        for pattern in party_patterns:
            matches = re.findall(pattern, text)
            entities["parties"].extend(matches)
        
        # Dates (various formats)
        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b'
        ]
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities["dates"].extend(matches)
        
        # Legal references (Section 5.2, Clause 11.1, Article III)
        ref_patterns = [
            r'(?:Section|Clause|Article|Paragraph|Exhibit|Schedule|Appendix)\s+(?:\d+(?:\.\d+)*|[IVXLCDM]+)',
        ]
        for pattern in ref_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities["references"].extend(matches)
        
        # Monetary amounts
        entities["amounts"] = TextProcessor.extract_monetary_amounts(text)
        
        # Email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        entities["emails"] = re.findall(email_pattern, text)
        
        # Phone numbers (US format)
        phone_pattern = r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b'
        phone_matches = re.findall(phone_pattern, text)
        entities["phone_numbers"] = ['-'.join(match) for match in phone_matches]
        
        # Deduplicate
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    @staticmethod
    def count_words(text: str) -> int:
        """Count words in text"""
        return len(text.split())
    
    @staticmethod
    def extract_numbers(text: str) -> List[str]:
        """Extract all numbers from text"""
        return re.findall(r'\d+', text)
    
    @staticmethod
    def extract_monetary_amounts(text: str) -> List[str]:
        """
        Extract monetary amounts from text
        
        Returns:
            List of monetary amounts (e.g., ['$1,000', '$2,500.00'])
        """
        # Match patterns like $1,000 or $1000.00 or USD 1,000
        patterns = [
            r'\$[\d,]+(?:\.\d{2})?',
            r'USD\s*[\d,]+(?:\.\d{2})?',
            r'EUR\s*[\d,]+(?:\.\d{2})?',
            r'GBP\s*[\d,]+(?:\.\d{2})?'
        ]
        
        amounts = []
        for pattern in patterns:
            amounts.extend(re.findall(pattern, text, re.IGNORECASE))
        
        return amounts
    
    @staticmethod
    def extract_durations(text: str) -> List[Dict[str, str]]:
        """
        Extract time durations (e.g., "6 months", "2 years")
        
        Returns:
            List of duration dictionaries with 'amount' and 'unit'
        """
        pattern = r'(\d+)\s*(day|week|month|year)s?'
        matches = re.findall(pattern, text, re.IGNORECASE)
        
        return [
            {"amount": m[0], "unit": m[1].lower()}
            for m in matches
        ]
    
    @staticmethod
    def extract_percentages(text: str) -> List[str]:
        """Extract percentages from text"""
        return re.findall(r'\d+(?:\.\d+)?%', text)
    
    # =========================================================================
    # TEXT CHUNKING FOR EMBEDDINGS
    # =========================================================================
    
    @staticmethod
    def chunk_text_for_embedding(text: str, 
                                 chunk_size: int = 512,
                                 overlap: int = 50) -> List[Dict[str, Any]]:
        """
        Chunk text with overlap for embedding models (preserves sentence boundaries)
        
        Args:
            text: Input text
            chunk_size: Maximum chunk size in words
            overlap: Number of words to overlap between chunks
        
        Returns:
            List of chunk dictionaries with metadata
        """
        sentences = TextProcessor.extract_sentences(text)
        chunks = []
        current_chunk = []
        current_length = 0
        start_sentence_idx = 0
        
        for i, sentence in enumerate(sentences):
            sentence_words = sentence.split()
            sentence_length = len(sentence_words)
            
            if current_length + sentence_length > chunk_size and current_chunk:
                # Save current chunk
                chunks.append({
                    "text": " ".join(current_chunk),
                    "start_sentence": start_sentence_idx,
                    "end_sentence": i - 1,
                    "word_count": current_length,
                    "chunk_id": len(chunks)
                })
                
                # Start new chunk with overlap
                overlap_sentences = current_chunk[-2:] if len(current_chunk) > 2 else current_chunk
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s.split()) for s in current_chunk)
                start_sentence_idx = max(0, i - len(overlap_sentences))
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunks.append({
                "text": " ".join(current_chunk),
                "start_sentence": start_sentence_idx,
                "end_sentence": len(sentences) - 1,
                "word_count": current_length,
                "chunk_id": len(chunks)
            })
        
        return chunks
    
    # =========================================================================
    # TEXT SIMILARITY & DEDUPLICATION
    # =========================================================================
    
    @staticmethod
    def text_similarity(text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts (0-1 scale)
        
        Args:
            text1: First text
            text2: Second text
        
        Returns:
            Similarity score (0.0 = completely different, 1.0 = identical)
        """
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    @staticmethod
    def deduplicate_clauses(clauses: List[str], threshold: float = 0.85) -> List[str]:
        """
        Remove near-duplicate clauses
        
        Args:
            clauses: List of clause texts
            threshold: Similarity threshold for deduplication (0.0-1.0)
        
        Returns:
            List of unique clauses
        """
        unique = []
        
        for clause in clauses:
            is_duplicate = any(
                TextProcessor.text_similarity(clause, existing) > threshold
                for existing in unique
            )
            if not is_duplicate:
                unique.append(clause)
        
        return unique
    
    # =========================================================================
    # LANGUAGE DETECTION
    # =========================================================================
    
    @staticmethod
    def detect_language(text: str) -> str:
        """
        Detect text language
        
        Args:
            text: Input text
        
        Returns:
            ISO 639-1 language code (e.g., 'en', 'es', 'fr')
        """
        if not LANGDETECT_AVAILABLE:
            return "en"  # Default to English
        
        try:
            # Use first 1000 chars for detection
            return detect(text[:1000])
        except LangDetectException:
            return "en"
    
    # =========================================================================
    # TEXT STATISTICS
    # =========================================================================
    
    @staticmethod
    def get_text_statistics(text: str) -> Dict[str, Any]:
        """
        Get comprehensive text statistics
        
        Returns:
            Dictionary with character count, word count, sentence count, etc.
        """
        sentences = TextProcessor.extract_sentences(text)
        paragraphs = TextProcessor.split_into_paragraphs(text)
        words = text.split()
        
        return {
            "character_count": len(text),
            "word_count": len(words),
            "sentence_count": len(sentences),
            "paragraph_count": len(paragraphs),
            "avg_words_per_sentence": len(words) / len(sentences) if sentences else 0,
            "avg_chars_per_word": len(text) / len(words) if words else 0,
            "language": TextProcessor.detect_language(text)
        }
    
    # =========================================================================
    # KEYWORD HIGHLIGHTING
    # =========================================================================
    
    @staticmethod
    def highlight_keywords(text: str, keywords: List[str], 
                          highlight_format: str = "**{}**") -> str:
        """
        Highlight keywords in text (for display purposes)
        
        Args:
            text: Input text
            keywords: List of keywords to highlight
            highlight_format: Format string with {} placeholder (default: Markdown bold)
        
        Returns:
            Text with highlighted keywords
        """
        for keyword in keywords:
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            text = pattern.sub(lambda m: highlight_format.format(m.group(0)), text)
        
        return text
    
    # =========================================================================
    # CLAUSE SEGMENTATION HELPERS
    # =========================================================================
    
    @staticmethod
    def extract_numbered_sections(text: str) -> List[Dict[str, Any]]:
        """
        Extract numbered sections/clauses (1.1, 1.2, Article 5, etc.)
        
        Returns:
            List of section dictionaries with number and text
        """
        patterns = [
            (r'(\d+\.\d+(?:\.\d+)*)\.\s*([^\n]{20,}?)(?=\n\s*\d+\.\d+|\n\n|$)', 'numbered'),
            (r'(Article\s+(?:\d+|[IVXLCDM]+))\.\s*([^\n]{20,}?)(?=\nArticle|\n\n|$)', 'article'),
            (r'(Section\s+(?:\d+|[IVXLCDM]+))\.\s*([^\n]{20,}?)(?=\nSection|\n\n|$)', 'section'),
            (r'(Clause\s+\d+(?:\.\d+)*)\.\s*([^\n]{20,}?)(?=\nClause|\n\n|$)', 'clause'),
        ]
        
        sections = []
        for pattern, section_type in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                sections.append({
                    "reference": match.group(1).strip(),
                    "text": match.group(2).strip(),
                    "type": section_type,
                    "start_pos": match.start(),
                    "end_pos": match.end()
                })
        
        # Sort by position
        sections.sort(key=lambda x: x['start_pos'])
        
        return sections
    
    @staticmethod
    def clean_legal_text(text: str) -> str:
        """
        Clean legal text by removing boilerplate artifacts
        
        Args:
            text: Input legal text
        
        Returns:
            Cleaned text
        """
        # Remove "Page X of Y" markers
        text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text, flags=re.IGNORECASE)
        
        # Remove "[Signature Page Follows]" type markers
        text = re.sub(r'\[.*?(?:Signature|Initial|Page).*?\]', '', text, flags=re.IGNORECASE)
        
        # Remove excessive underscores (signature lines)
        text = re.sub(r'_{3,}', '', text)
        
        # Remove "CONFIDENTIAL" watermarks
        text = re.sub(r'\b(CONFIDENTIAL|DRAFT|INTERNAL USE ONLY)\b', '', text, flags=re.IGNORECASE)
        
        # Clean up resulting whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        return text.strip()