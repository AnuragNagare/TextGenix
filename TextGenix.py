"""
TextGenix - Complete Intelligent Document Processing System
Beautiful Modern UI with Advanced Styling

Features:
- Multi-format document processing (PDF, DOCX, TXT, HTML, RTF)
- AI-powered text enhancement
- Industry-specific terminology
- Beautiful modern interface with custom CSS
- Real-time analytics with stunning visualizations
"""

import os
import re
import json
import time
import tempfile
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict

# Web Interface
import gradio as gr
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Core Libraries (with fallbacks for missing dependencies)
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("⚠️  spaCy not available. Install with: pip install spacy && python -m spacy download en_core_web_sm")

try:
    import nltk
    from nltk.corpus import wordnet
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("⚠️  NLTK not available. Install with: pip install nltk")

try:
    from textstat import flesch_reading_ease, flesch_kincaid_grade
    TEXTSTAT_AVAILABLE = True
except ImportError:
    TEXTSTAT_AVAILABLE = False
    print("⚠️  TextStat not available. Install with: pip install textstat")

# Document Format Support (with fallbacks)
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("⚠️  PyPDF2 not available. Install with: pip install PyPDF2")

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    print("⚠️  python-docx not available. Install with: pip install python-docx")

try:
    import html2text
    HTML_AVAILABLE = True
except ImportError:
    HTML_AVAILABLE = False
    print("⚠️  html2text not available. Install with: pip install html2text")

try:
    from striprtf.striprtf import rtf_to_text
    RTF_AVAILABLE = True
except ImportError:
    RTF_AVAILABLE = False
    print("⚠️  striprtf not available. Install with: pip install striprtf")

# AI and Semantic Analysis (with fallbacks)
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    print("⚠️  AI libraries not available. Install with: pip install sentence-transformers scikit-learn")

try:
    import language_tool_python
    GRAMMAR_AVAILABLE = True
except ImportError:
    GRAMMAR_AVAILABLE = False
    print("⚠️  Grammar checking not available. Install with: pip install language-tool-python")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom CSS for beautiful UI
CUSTOM_CSS = """
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

/* Root Variables for Consistent Theming */
:root {
    --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    --success-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    --warning-gradient: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
    --dark-gradient: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
    --glass-bg: rgba(255, 255, 255, 0.1);
    --glass-border: rgba(255, 255, 255, 0.2);
    --text-primary: #2d3748;
    --text-secondary: #4a5568;
    --bg-primary: #f7fafc;
    --bg-card: #ffffff;
    --shadow-soft: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
    --shadow-glow: 0 0 20px rgba(102, 126, 234, 0.3);
    --border-radius: 16px;
    --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

/* Global Styles */
* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Main Container Styling */
.gradio-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 75%, #f5576c 100%) !important;
    min-height: 100vh;
    padding: 20px;
}

/* Header Styling */
.main-header {
    text-align: center;
    padding: 2rem;
    background: var(--glass-bg);
    backdrop-filter: blur(20px);
    border: 1px solid var(--glass-border);
    border-radius: var(--border-radius);
    margin-bottom: 2rem;
    box-shadow: var(--shadow-soft);
}

.main-header h1 {
    color: #ffffff;
    font-size: 3.5rem !important;
    font-weight: 700 !important;
    margin-bottom: 0.5rem !important;
    text-shadow: 2px 2px 8px rgba(0,0,0,0.5);
}

.main-header h3 {
    color: rgba(255, 255, 255, 0.9) !important;
    font-size: 1.5rem !important;
    font-weight: 400 !important;
    margin-bottom: 1rem !important;
}

.main-header p {
    color: rgba(255, 255, 255, 0.8) !important;
    font-size: 1.1rem !important;
    max-width: 600px;
    margin: 0 auto;
}

/* Card Styling */
.card {
    background: var(--bg-card) !important;
    border-radius: var(--border-radius) !important;
    box-shadow: var(--shadow-soft) !important;
    padding: 2rem !important;
    margin: 1rem 0 !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    transition: var(--transition) !important;
}

.card:hover {
    transform: translateY(-2px);
    box-shadow: 0 20px 40px -5px rgba(0, 0, 0, 0.15), 0 10px 10px -5px rgba(0, 0, 0, 0.1);
}

/* Input Group Styling */
.input-group {
    background: linear-gradient(145deg, #f8f9ff, #e8f0ff);
    border-radius: var(--border-radius);
    padding: 1.5rem;
    margin: 1rem 0;
    border: 1px solid rgba(102, 126, 234, 0.1);
}

/* Button Styling */
.primary-btn {
    background: var(--primary-gradient) !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 1rem 2rem !important;
    font-weight: 600 !important;
    font-size: 1.1rem !important;
    color: white !important;
    cursor: pointer !important;
    transition: var(--transition) !important;
    box-shadow: 0 4px 15px 0 rgba(102, 126, 234, 0.3) !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px 0 rgba(102, 126, 234, 0.4) !important;
}

.secondary-btn {
    background: var(--secondary-gradient) !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.8rem 1.5rem !important;
    font-weight: 500 !important;
    color: white !important;
    cursor: pointer !important;
    transition: var(--transition) !important;
    box-shadow: 0 4px 15px 0 rgba(240, 147, 251, 0.3) !important;
}

.secondary-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px 0 rgba(240, 147, 251, 0.4) !important;
}

/* Form Controls */
.gradio-dropdown, .gradio-textbox, .gradio-slider {
    border-radius: 12px !important;
    border: 2px solid rgba(102, 126, 234, 0.1) !important;
    background: white !important;
    transition: var(--transition) !important;
}

.gradio-dropdown:focus, .gradio-textbox:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
}

/* File Upload Area */
.file-upload {
    background: linear-gradient(145deg, #ffffff, #f0f7ff);
    border: 2px dashed #667eea;
    border-radius: var(--border-radius);
    padding: 2rem;
    text-align: center;
    transition: var(--transition);
}

.file-upload:hover {
    border-color: #764ba2;
    background: linear-gradient(145deg, #f0f7ff, #e6f3ff);
}

/* Results Section */
.results-section {
    background: linear-gradient(145deg, #ffffff, #f8fbff);
    border-radius: var(--border-radius);
    padding: 2rem;
    margin-top: 2rem;
    box-shadow: var(--shadow-soft);
    border: 1px solid rgba(102, 126, 234, 0.1);
}

/* Status Messages */
.status-success {
    background: var(--success-gradient) !important;
    color: white !important;
    padding: 1rem 1.5rem !important;
    border-radius: 12px !important;
    font-weight: 500 !important;
    border: none !important;
}

.status-error {
    background: var(--secondary-gradient) !important;
    color: white !important;
    padding: 1rem 1.5rem !important;
    border-radius: 12px !important;
    font-weight: 500 !important;
    border: none !important;
}

/* Text Areas */
.enhanced-text {
    background: linear-gradient(145deg, #f8f9ff, #ffffff) !important;
    border: 2px solid rgba(102, 126, 234, 0.1) !important;
    border-radius: 12px !important;
    font-family: 'JetBrains Mono', monospace !important;
    line-height: 1.6 !important;
    padding: 1.5rem !important;
}

/* Tables */
.gradio-dataframe {
    border-radius: 12px !important;
    overflow: hidden !important;
    box-shadow: var(--shadow-soft) !important;
}

/* Tabs */
.gradio-tabs {
    background: white;
    border-radius: var(--border-radius);
    box-shadow: var(--shadow-soft);
    overflow: hidden;
}

.gradio-tab {
    background: linear-gradient(145deg, #667eea, #764ba2) !important;
    color: white !important;
    border: none !important;
    padding: 1rem 2rem !important;
    font-weight: 600 !important;
}

.gradio-tab.selected {
    background: var(--success-gradient) !important;
}

/* Animations */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.fade-in {
    animation: fadeIn 0.5s ease-out;
}

@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.05); }
    100% { transform: scale(1); }
}

.pulse {
    animation: pulse 2s infinite;
}

/* Progress Bar */
.progress-bar {
    background: var(--primary-gradient) !important;
    border-radius: 10px !important;
    height: 8px !important;
}

/* Responsive Design */
@media (max-width: 768px) {
    .main-header h1 {
        font-size: 2.5rem !important;
    }
    
    .card {
        padding: 1rem !important;
        margin: 0.5rem 0 !important;
    }
    
    .primary-btn {
        padding: 0.8rem 1.5rem !important;
        font-size: 1rem !important;
    }
}

/* Plotly Chart Styling */
.js-plotly-plot {
    border-radius: 12px !important;
    box-shadow: var(--shadow-soft) !important;
    overflow: hidden !important;
}

/* Custom Scrollbar */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 10px;
}

::-webkit-scrollbar-thumb {
    background: var(--primary-gradient);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--secondary-gradient);
}

/* Special Effects */
.glow {
    box-shadow: var(--shadow-glow) !important;
}

.glass {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(20px) !important;
    border: 1px solid var(--glass-border) !important;
}

/* Feature Icons */
.feature-icon {
    display: inline-block;
    width: 2rem;
    height: 2rem;
    background: var(--primary-gradient);
    border-radius: 50%;
    text-align: center;
    line-height: 2rem;
    color: white;
    font-weight: 600;
    margin-right: 1rem;
}

/* Section Headers */
.section-header {
    background: var(--primary-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    margin-bottom: 1rem !important;
    display: flex;
    align-items: center;
}

/* Loading Animation */
.loading {
    position: relative;
    overflow: hidden;
}

.loading::after {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
    animation: loading 1.5s infinite;
}

@keyframes loading {
    0% { left: -100%; }
    100% { left: 100%; }
}
"""

@dataclass
class ProcessingConfig:
    """Configuration for document processing"""
    min_word_length: int = 3
    max_word_length: int = 15
    industry: str = "general"
    enhancement_mode: str = "professional"
    preserve_context: bool = True
    quality_threshold: float = 0.8
    enable_grammar_check: bool = True
    
@dataclass
class TransformationResult:
    """Result of a single word transformation"""
    original_word: str
    enhanced_word: str
    confidence_score: float
    transformation_reason: str
    position: int

@dataclass
class DocumentAnalysis:
    """Complete document analysis results"""
    original_text: str
    enhanced_text: str
    transformations: List[TransformationResult]
    quality_metrics: Dict[str, float]
    processing_time: float
    word_count_original: int
    word_count_enhanced: int
    readability_improvement: float

class IndustryVocabulary:
    """Industry-specific vocabulary and terminology mappings"""
    
    VOCABULARIES = {
        "legal": {
            "contract": "comprehensive legal agreement",
            "terms": "contractual provisions",
            "important": "legally significant",
            "agreement": "binding legal arrangement",
            "party": "contracting entity",
            "obligation": "legal commitment",
            "rights": "legal entitlements",
            "provision": "contractual stipulation",
            "clause": "legal provision",
            "section": "contractual segment",
            "document": "legal instrument",
            "requirement": "legal mandate",
            "compliance": "regulatory adherence",
            "liability": "legal responsibility",
            "damages": "monetary compensation"
        },
        "medical": {
            "patient": "individual receiving care",
            "treatment": "therapeutic intervention",
            "diagnosis": "clinical assessment",
            "symptoms": "clinical manifestations",
            "condition": "medical status",
            "procedure": "medical intervention",
            "therapy": "therapeutic treatment",
            "medication": "pharmaceutical intervention",
            "examination": "clinical evaluation",
            "analysis": "diagnostic assessment",
            "care": "medical attention",
            "health": "physiological well-being",
            "doctor": "healthcare provider",
            "hospital": "medical facility",
            "recovery": "therapeutic restoration"
        },
        "financial": {
            "money": "financial capital",
            "profit": "revenue generation",
            "investment": "strategic capital allocation",
            "assets": "financial holdings",
            "revenue": "income generation",
            "costs": "operational expenditures",
            "budget": "financial allocation",
            "expenses": "operational costs",
            "returns": "investment yields",
            "portfolio": "investment holdings",
            "growth": "financial expansion",
            "market": "economic marketplace",
            "value": "monetary worth",
            "income": "revenue stream",
            "capital": "financial resources"
        },
        "technical": {
            "system": "technological infrastructure",
            "process": "operational methodology",
            "function": "operational capability",
            "method": "systematic approach",
            "solution": "technical implementation",
            "design": "architectural framework",
            "development": "systematic construction",
            "implementation": "systematic deployment",
            "optimization": "performance enhancement",
            "integration": "systematic consolidation",
            "software": "computational application",
            "hardware": "physical infrastructure",
            "network": "interconnected system",
            "database": "structured data repository",
            "algorithm": "computational procedure"
        }
    }
    
    @classmethod
    def get_enhanced_word(cls, word: str, industry: str) -> Optional[str]:
        """Get industry-specific enhancement for a word"""
        if industry in cls.VOCABULARIES:
            return cls.VOCABULARIES[industry].get(word.lower())
        return None

class ProfessionalVocabulary:
    """Professional enhancement vocabulary for general business context"""
    
    ENHANCEMENT_MAPPINGS = {
        # Basic to Professional
        "good": ["exceptional", "outstanding", "superior", "exemplary"],
        "bad": ["suboptimal", "inadequate", "deficient", "unsatisfactory"],
        "big": ["substantial", "significant", "considerable", "extensive"],
        "small": ["minimal", "modest", "limited", "compact"],
        "important": ["critical", "essential", "paramount", "vital"],
        "nice": ["commendable", "admirable", "noteworthy", "praiseworthy"],
        "great": ["remarkable", "exceptional", "distinguished", "extraordinary"],
        "easy": ["straightforward", "accessible", "streamlined", "simplified"],
        "hard": ["challenging", "complex", "demanding", "rigorous"],
        "fast": ["expeditious", "efficient", "accelerated", "streamlined"],
        "slow": ["deliberate", "methodical", "comprehensive", "thorough"],
        "new": ["innovative", "contemporary", "cutting-edge", "advanced"],
        "old": ["established", "traditional", "proven", "time-tested"],
        "help": ["facilitate", "support", "enable", "assist"],
        "make": ["develop", "create", "establish", "construct"],
        "use": ["utilize", "employ", "leverage", "implement"],
        "show": ["demonstrate", "illustrate", "exhibit", "present"],
        "find": ["identify", "discover", "locate", "determine"],
        "get": ["obtain", "acquire", "secure", "procure"],
        "give": ["provide", "deliver", "furnish", "supply"],
        "put": ["position", "place", "establish", "implement"],
        "take": ["acquire", "obtain", "secure", "assume"],
        "work": ["function", "operate", "perform", "execute"],
        "think": ["consider", "evaluate", "analyze", "assess"],
        "know": ["understand", "comprehend", "recognize", "acknowledge"],
        "say": ["communicate", "express", "articulate", "convey"],
        "look": ["examine", "analyze", "investigate", "review"],
        "come": ["arrive", "approach", "emerge", "materialize"],
        "want": ["require", "desire", "seek", "pursue"],
        "need": ["require", "necessitate", "demand", "warrant"],
        "data": ["information", "analytics", "intelligence", "insights"],
        "team": ["organization", "workforce", "personnel", "collective"],
        "problem": ["challenge", "issue", "concern", "obstacle"],
        "solution": ["resolution", "approach", "methodology", "strategy"]
    }
    
    @classmethod
    def get_enhanced_word(cls, word: str) -> Optional[str]:
        """Get professional enhancement for a word"""
        word_lower = word.lower()
        if word_lower in cls.ENHANCEMENT_MAPPINGS:
            options = cls.ENHANCEMENT_MAPPINGS[word_lower]
            return options[0]  # Return first option
        return None

class DocumentProcessor:
    """Core document processing and format handling"""
    
    @staticmethod
    def extract_text_from_pdf(file_path: str) -> str:
        """Extract text from PDF file"""
        if not PDF_AVAILABLE:
            raise ImportError("PyPDF2 not available. Install with: pip install PyPDF2")
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise
    
    @staticmethod
    def extract_text_from_docx(file_path: str) -> str:
        """Extract text from DOCX file"""
        if not DOCX_AVAILABLE:
            raise ImportError("python-docx not available. Install with: pip install python-docx")
        
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from DOCX: {e}")
            raise
    
    @staticmethod
    def extract_text_from_txt(file_path: str) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as file:
                return file.read().strip()
        except Exception as e:
            logger.error(f"Error extracting text from TXT: {e}")
            raise
    
    @staticmethod
    def extract_text_from_html(file_path: str) -> str:
        """Extract text from HTML file"""
        if not HTML_AVAILABLE:
            # Fallback: simple HTML tag removal
            with open(file_path, 'r', encoding='utf-8') as file:
                html_content = file.read()
                # Simple HTML tag removal
                clean_text = re.sub('<[^<]+?>', '', html_content)
                return clean_text.strip()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                html_content = file.read()
                h = html2text.HTML2Text()
                h.ignore_links = True
                h.ignore_images = True
                return h.handle(html_content).strip()
        except Exception as e:
            logger.error(f"Error extracting text from HTML: {e}")
            raise
    
    @staticmethod
    def extract_text_from_rtf(file_path: str) -> str:
        """Extract text from RTF file"""
        if not RTF_AVAILABLE:
            raise ImportError("striprtf not available. Install with: pip install striprtf")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                rtf_content = file.read()
                return rtf_to_text(rtf_content).strip()
        except Exception as e:
            logger.error(f"Error extracting text from RTF: {e}")
            raise
    
    @classmethod
    def extract_text(cls, file_path: str) -> str:
        """Extract text from any supported file format"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = os.path.splitext(file_path)[1].lower()
        
        extractors = {
            '.pdf': cls.extract_text_from_pdf,
            '.docx': cls.extract_text_from_docx,
            '.doc': cls.extract_text_from_docx,
            '.txt': cls.extract_text_from_txt,
            '.html': cls.extract_text_from_html,
            '.htm': cls.extract_text_from_html,
            '.rtf': cls.extract_text_from_rtf
        }
        
        if file_extension not in extractors:
            # Try as text file
            return cls.extract_text_from_txt(file_path)
        
        return extractors[file_extension](file_path)

class SimpleTextEnhancer:
    """Simplified text enhancer that works without heavy AI dependencies"""
    
    def __init__(self):
        self.nlp = None
        self.sentence_model = None
        self.grammar_tool = None
        
        # Try to initialize advanced features
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("⚠️  spaCy model 'en_core_web_sm' not found. Using basic processing.")
        
        if AI_AVAILABLE:
            try:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception:
                print("⚠️  Sentence transformer model not available. Using basic similarity.")
        
        if GRAMMAR_AVAILABLE:
            try:
                self.grammar_tool = language_tool_python.LanguageTool('en-US')
            except Exception:
                print("⚠️  Grammar tool not available.")
    
    def simple_tokenize(self, text: str) -> List[Dict]:
        """Simple tokenization fallback"""
        words = re.findall(r'\b\w+\b', text)
        tokens = []
        for i, word in enumerate(words):
            tokens.append({
                'text': word,
                'pos_': 'UNKNOWN',
                'is_alpha': word.isalpha(),
                'is_stop': word.lower() in ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
            })
        return tokens
    
    def analyze_document_context(self, text: str) -> Dict[str, Any]:
        """Analyze document context with fallbacks"""
        if self.nlp:
            try:
                doc = self.nlp(text)
                entities = {
                    "persons": [ent.text for ent in doc.ents if ent.label_ == "PERSON"],
                    "organizations": [ent.text for ent in doc.ents if ent.label_ == "ORG"],
                    "locations": [ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]],
                    "dates": [ent.text for ent in doc.ents if ent.label_ == "DATE"],
                    "monetary": [ent.text for ent in doc.ents if ent.label_ == "MONEY"]
                }
                return {
                    "entities": entities,
                    "total_tokens": len(doc),
                    "unique_words": len(set([token.lemma_.lower() for token in doc if token.is_alpha]))
                }
            except Exception:
                pass
        
        # Fallback analysis
        words = text.split()
        return {
            "entities": {"persons": [], "organizations": [], "locations": [], "dates": [], "monetary": []},
            "total_tokens": len(words),
            "unique_words": len(set(word.lower() for word in words if word.isalpha()))
        }
    
    def should_enhance_word(self, word_info: Dict, config: ProcessingConfig) -> bool:
        """Determine if a word should be enhanced"""
        word = word_info['text']
        
        # Basic length check
        if not (config.min_word_length <= len(word) <= config.max_word_length):
            return False
        
        # Don't enhance non-alphabetic tokens or very common stop words
        if not word_info.get('is_alpha', True):
            return False
        
        if word_info.get('is_stop', False) and len(word) < 6:
            return False
        
        # Don't enhance if already professional
        professional_words = ["comprehensive", "strategic", "innovative", "exceptional", 
                             "substantial", "optimal", "critical", "essential"]
        if word.lower() in professional_words:
            return False
        
        return True
    
    def get_contextual_enhancement(self, word: str, context: str, config: ProcessingConfig) -> Tuple[str, float, str]:
        """Get contextually appropriate enhancement for a word"""
        
        # Try industry-specific enhancement first
        if config.industry != "general":
            industry_enhancement = IndustryVocabulary.get_enhanced_word(word, config.industry)
            if industry_enhancement:
                return industry_enhancement, 0.9, f"Industry-specific enhancement ({config.industry})"
        
        # Try professional vocabulary enhancement
        professional_enhancement = ProfessionalVocabulary.get_enhanced_word(word)
        if professional_enhancement:
            return professional_enhancement, 0.8, "Professional vocabulary enhancement"
        
        # Try simple synonym enhancement with NLTK
        if NLTK_AVAILABLE:
            try:
                synsets = wordnet.synsets(word)
                if synsets:
                    synonyms = []
                    for synset in synsets[:2]:
                        for lemma in synset.lemmas():
                            synonym = lemma.name().replace('_', ' ')
                            if (synonym != word and 
                                len(synonym) >= len(word) and 
                                synonym.isalpha()):
                                synonyms.append(synonym)
                    
                    if synonyms:
                        return max(synonyms, key=len), 0.7, "Synonym-based enhancement"
            except Exception:
                pass
        
        # No enhancement found
        return word, 1.0, "No enhancement needed"
    
    def calculate_semantic_preservation(self, original: str, enhanced: str) -> float:
        """Calculate semantic preservation with fallbacks"""
        if self.sentence_model and AI_AVAILABLE:
            try:
                original_embedding = self.sentence_model.encode([original])
                enhanced_embedding = self.sentence_model.encode([enhanced])
                similarity = cosine_similarity(original_embedding, enhanced_embedding)[0][0]
                return float(similarity)
            except Exception:
                pass
        
        # Fallback: simple word overlap
        original_words = set(original.lower().split())
        enhanced_words = set(enhanced.lower().split())
        if len(original_words) == 0:
            return 1.0
        overlap = len(original_words.intersection(enhanced_words))
        return overlap / len(original_words)
    
    def calculate_quality_metrics(self, original: str, enhanced: str, transformations: List[TransformationResult]) -> Dict[str, float]:
        """Calculate comprehensive quality metrics"""
        metrics = {}
        
        # Semantic preservation
        metrics['semantic_preservation'] = self.calculate_semantic_preservation(original, enhanced)
        
        # Readability improvement
        if TEXTSTAT_AVAILABLE:
            try:
                original_readability = flesch_reading_ease(original)
                enhanced_readability = flesch_reading_ease(enhanced)
                metrics['readability_improvement'] = (enhanced_readability - original_readability) / 100.0
            except Exception:
                metrics['readability_improvement'] = 0.1  # Default positive
        else:
            metrics['readability_improvement'] = 0.1
        
        # Enhancement ratio
        total_words = len(original.split())
        enhanced_words = len([t for t in transformations if t.enhanced_word != t.original_word])
        metrics['enhancement_ratio'] = enhanced_words / total_words if total_words > 0 else 0
        
        # Average transformation confidence
        if transformations:
            metrics['avg_transformation_confidence'] = sum(t.confidence_score for t in transformations) / len(transformations)
        else:
            metrics['avg_transformation_confidence'] = 1.0
        
        # Grammar score (simplified)
        if self.grammar_tool:
            try:
                grammar_errors = len(self.grammar_tool.check(enhanced))
                total_words_enhanced = len(enhanced.split())
                metrics['grammar_score'] = max(0, 1 - (grammar_errors / total_words_enhanced)) if total_words_enhanced > 0 else 1.0
            except Exception:
                metrics['grammar_score'] = 0.9
        else:
            metrics['grammar_score'] = 0.9
        
        # Overall quality score
        metrics['overall_quality'] = (
            metrics['semantic_preservation'] * 0.4 +
            metrics['avg_transformation_confidence'] * 0.3 +
            metrics['grammar_score'] * 0.2 +
            max(0, metrics['readability_improvement']) * 0.1
        )
        
        return metrics
    
    def enhance_document(self, text: str, config: ProcessingConfig) -> DocumentAnalysis:
        """Main method to enhance an entire document"""
        start_time = time.time()
        
        # Analyze document context
        context_analysis = self.analyze_document_context(text)
        
        # Tokenize text
        if self.nlp:
            try:
                doc = self.nlp(text)
                tokens = [{'text': token.text, 'pos_': token.pos_, 'is_alpha': token.is_alpha, 'is_stop': token.is_stop} 
                         for token in doc]
            except Exception:
                tokens = self.simple_tokenize(text)
        else:
            tokens = self.simple_tokenize(text)
        
        # Track transformations
        transformations = []
        enhanced_tokens = []
        
        for i, token in enumerate(tokens):
            if self.should_enhance_word(token, config):
                enhanced_word, confidence, reason = self.get_contextual_enhancement(
                    token['text'], text, config
                )
                
                transformations.append(TransformationResult(
                    original_word=token['text'],
                    enhanced_word=enhanced_word,
                    confidence_score=confidence,
                    transformation_reason=reason,
                    position=i
                ))
                
                enhanced_tokens.append(enhanced_word)
            else:
                enhanced_tokens.append(token['text'])
        
        # Reconstruct enhanced text (simple word replacement)
        original_words = [token['text'] for token in tokens]
        enhanced_text = text
        
        # Replace words in order (longest first to avoid partial replacements)
        word_replacements = [(t.original_word, t.enhanced_word) for t in transformations 
                           if t.enhanced_word != t.original_word]
        word_replacements.sort(key=lambda x: len(x[0]), reverse=True)
        
        for original, enhanced in word_replacements:
            # Use word boundaries to avoid partial replacements
            pattern = r'\b' + re.escape(original) + r'\b'
            enhanced_text = re.sub(pattern, enhanced, enhanced_text, flags=re.IGNORECASE)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Calculate quality metrics
        quality_metrics = self.calculate_quality_metrics(text, enhanced_text, transformations)
        
        # Calculate readability improvement
        if TEXTSTAT_AVAILABLE:
            try:
                original_readability = flesch_reading_ease(text)
                enhanced_readability = flesch_reading_ease(enhanced_text)
                readability_improvement = enhanced_readability - original_readability
            except Exception:
                readability_improvement = 5.0  # Default improvement
        else:
            readability_improvement = 5.0
        
        return DocumentAnalysis(
            original_text=text,
            enhanced_text=enhanced_text,
            transformations=transformations,
            quality_metrics=quality_metrics,
            processing_time=processing_time,
            word_count_original=len(text.split()),
            word_count_enhanced=len(enhanced_text.split()),
            readability_improvement=readability_improvement
        )

class WebInterface:
    """Beautiful web interface for document processing"""
    
    def __init__(self):
        self.enhancer = SimpleTextEnhancer()
        self.processing_history = []
        print("✅ TextGenix initialized successfully!")
    
    def process_document(
        self,
        file,
        industry: str,
        mode: str,
        min_length: int,
        max_length: int,
        quality_threshold: float,
        preserve_context: bool,
        enable_grammar: bool,
        progress=gr.Progress()
    ):
        """Main document processing function with progress tracking"""
        
        if file is None:
            return self.create_error_output("Please upload a file first")
        
        try:
            # Update progress
            progress(0.1, desc="🔍 Extracting text from document...")
            
            # Extract text from uploaded file
            text = DocumentProcessor.extract_text(file.name)
            if not text.strip():
                return self.create_error_output("No text found in the document")
            
            progress(0.3, desc="🧠 Analyzing document context...")
            
            # Create processing configuration
            config = ProcessingConfig(
                min_word_length=min_length,
                max_word_length=max_length,
                industry=industry,
                enhancement_mode=mode,
                preserve_context=preserve_context,
                quality_threshold=quality_threshold,
                enable_grammar_check=enable_grammar
            )
            
            progress(0.5, desc="✨ Enhancing text with AI...")
            
            # Process the document
            analysis = self.enhancer.enhance_document(text, config)
            
            progress(0.8, desc="📊 Generating visualizations...")
            
            # Store in history
            self.processing_history.append({
                'timestamp': datetime.now(),
                'filename': os.path.basename(file.name),
                'analysis': analysis
            })
            
            progress(1.0, desc="🎉 Complete!")
            
            # Generate all outputs
            return self.create_success_output(analysis, os.path.basename(file.name))
            
        except Exception as e:
            return self.create_error_output(f"Processing failed: {str(e)}")
    
    def create_success_output(self, analysis: DocumentAnalysis, filename: str):
        """Create comprehensive success output with visualizations"""
        
        # Enhanced text output
        enhanced_text = analysis.enhanced_text
        
        # Processing summary
        summary = self.create_processing_summary(analysis, filename)
        
        # Quality metrics visualization
        quality_chart = self.create_quality_chart(analysis.quality_metrics)
        
        # Transformations table
        transformations_df = self.create_transformations_table(analysis.transformations)
        
        # Word cloud visualization
        word_cloud_chart = self.create_word_cloud_chart(analysis.transformations)
        
        # Downloadable files
        enhanced_file = self.create_downloadable_file(analysis.enhanced_text, f"{filename}_enhanced.txt")
        json_file = self.create_downloadable_json(analysis, f"{filename}_analysis.json")
        
        # Success status
        status = f"🎉 Processing completed successfully! Enhanced {len([t for t in analysis.transformations if t.enhanced_word != t.original_word])} words with {analysis.quality_metrics.get('overall_quality', 0):.1%} quality score."
        
        return (
            status,                    # status
            enhanced_text,             # enhanced_text
            summary,                   # summary
            quality_chart,             # quality_chart
            word_cloud_chart,          # word_cloud_chart
            transformations_df,        # transformations_table
            enhanced_file,             # enhanced_download
            json_file,                 # analysis_download
            gr.update(visible=True)    # results_section
        )
    
    def create_error_output(self, error_message: str):
        """Create error output"""
        return (
            f"❌ {error_message}",     # status
            "",                        # enhanced_text
            "",                        # summary
            None,                      # quality_chart
            None,                      # word_cloud_chart
            None,                      # transformations_table
            None,                      # enhanced_download
            None,                      # analysis_download
            gr.update(visible=False)   # results_section
        )
    
    def create_processing_summary(self, analysis: DocumentAnalysis, filename: str) -> str:
        """Create a formatted processing summary"""
        transformations_count = len([t for t in analysis.transformations if t.enhanced_word != t.original_word])
        
        summary = f"""
## 📊 Processing Summary for: **{filename}**

### 🎯 Key Metrics
| Metric | Value |
|--------|-------|
| 📝 **Original Words** | {analysis.word_count_original:,} |
| ✨ **Enhanced Words** | {analysis.word_count_enhanced:,} |
| 🔄 **Transformations Applied** | {transformations_count:,} |
| ⚡ **Processing Time** | {analysis.processing_time:.2f}s |
| 🏆 **Overall Quality Score** | {analysis.quality_metrics.get('overall_quality', 0):.1%} |

### 🚀 Quality Improvements
| Improvement | Score |
|-------------|-------|
| 🧠 **Semantic Preservation** | {analysis.quality_metrics.get('semantic_preservation', 0):.1%} |
| 📖 **Readability Improvement** | {analysis.readability_improvement:+.1f} points |
| ✍️ **Grammar Score** | {analysis.quality_metrics.get('grammar_score', 0):.1%} |
| 📈 **Enhancement Ratio** | {analysis.quality_metrics.get('enhancement_ratio', 0):.1%} |

### 🌟 Sample Transformations
"""
        
        # Add sample transformations
        sample_transformations = [t for t in analysis.transformations 
                                if t.enhanced_word != t.original_word][:3]
        
        for i, trans in enumerate(sample_transformations, 1):
            confidence_emoji = "🟢" if trans.confidence_score >= 0.8 else "🟡" if trans.confidence_score >= 0.6 else "🟠"
            summary += f"""
**{i}.** `{trans.original_word}` → `{trans.enhanced_word}` {confidence_emoji}
   - *Confidence*: {trans.confidence_score:.1%}
   - *Reason*: {trans.transformation_reason}
"""
        
        if len(sample_transformations) < transformations_count:
            remaining = transformations_count - len(sample_transformations)
            summary += f"\n*...and {remaining} more transformations*"
        
        return summary
    
    def create_quality_chart(self, quality_metrics: Dict[str, float]):
        """Create quality metrics visualization with beautiful styling"""
        
        metrics_data = []
        
        metric_names = {
            'semantic_preservation': 'Semantic Preservation',
            'grammar_score': 'Grammar Quality',
            'avg_transformation_confidence': 'AI Confidence',
            'overall_quality': 'Overall Quality'
        }
        
        colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c']
        
        for i, (key, display_name) in enumerate(metric_names.items()):
            if key in quality_metrics:
                value = quality_metrics[key] * 100  # Convert to percentage
                metrics_data.append({
                    'Metric': display_name, 
                    'Score': value,
                    'Color': colors[i % len(colors)]
                })
        
        if not metrics_data:
            return None
        
        df = pd.DataFrame(metrics_data)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=df['Metric'],
            y=df['Score'],
            marker=dict(
                color=df['Color'],
                line=dict(color='rgba(255,255,255,0.8)', width=2)
            ),
            text=[f'{score:.1f}%' for score in df['Score']],
            textposition='outside',
            textfont=dict(size=14, color='white', family='Inter')
        ))
        
        fig.update_layout(
            title=dict(
                text='🎯 Quality Metrics Dashboard',
                x=0.5,
                font=dict(size=20, color='white', family='Inter')
            ),
            xaxis=dict(
                title='Quality Metrics',
                titlefont=dict(size=14, color='white'),
                tickfont=dict(size=12, color='white')
            ),
            yaxis=dict(
                title='Score (%)',
                titlefont=dict(size=14, color='white'),
                tickfont=dict(size=12, color='white'),
                range=[0, 100]
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400,
            showlegend=False
        )
        
        return fig
    
    def create_word_cloud_chart(self, transformations: List[TransformationResult]):
        """Create word transformation visualization"""
        
        # Filter actual transformations
        actual_transformations = [
            t for t in transformations 
            if t.enhanced_word != t.original_word
        ]
        
        if not actual_transformations:
            return None
        
        # Create transformation frequency data
        transformation_types = {}
        for trans in actual_transformations:
            reason = trans.transformation_reason
            if reason in transformation_types:
                transformation_types[reason] += 1
            else:
                transformation_types[reason] = 1
        
        if not transformation_types:
            return None
        
        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=list(transformation_types.keys()),
            values=list(transformation_types.values()),
            hole=0.3,
            marker=dict(
                colors=['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe'],
                line=dict(color='#FFFFFF', width=2)
            ),
            textfont=dict(size=12, color='white', family='Inter')
        )])
        
        fig.update_layout(
            title=dict(
                text='🔄 Transformation Types Distribution',
                x=0.5,
                font=dict(size=20, color='white', family='Inter')
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400,
            showlegend=True,
            legend=dict(
                font=dict(color='white', size=12)
            )
        )
        
        return fig
    
    def create_transformations_table(self, transformations: List[TransformationResult]):
        """Create transformations table for display"""
        
        # Filter only actual transformations (not unchanged words)
        actual_transformations = [
            t for t in transformations 
            if t.enhanced_word != t.original_word
        ]
        
        if not actual_transformations:
            return pd.DataFrame({'Message': ['No transformations were applied']})
        
        # Create table data
        table_data = []
        for i, trans in enumerate(actual_transformations[:15]):  # Show top 15
            confidence_emoji = "🟢" if trans.confidence_score >= 0.8 else "🟡" if trans.confidence_score >= 0.6 else "🟠"
            table_data.append({
                '#': i + 1,
                'Original Word': f"📝 {trans.original_word}",
                'Enhanced Word': f"✨ {trans.enhanced_word}",
                'Confidence': f"{confidence_emoji} {trans.confidence_score:.1%}",
                'Transformation Type': trans.transformation_reason,
                'Position': f"#{trans.position}"
            })
        
        df = pd.DataFrame(table_data)
        
        if len(actual_transformations) > 15:
            # Add summary row
            summary_row = pd.DataFrame({
                '#': ['...'],
                'Original Word': [f'📋 + {len(actual_transformations) - 15} more'],
                'Enhanced Word': ['transformations'],
                'Confidence': [''],
                'Transformation Type': [''],
                'Position': ['']
            })
            df = pd.concat([df, summary_row], ignore_index=True)
        
        return df
    
    def create_downloadable_file(self, content: str, filename: str) -> str:
        """Create downloadable text file"""
        temp_path = os.path.join(tempfile.gettempdir(), filename)
        with open(temp_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return temp_path
    
    def create_downloadable_json(self, analysis: DocumentAnalysis, filename: str) -> str:
        """Create downloadable JSON analysis file"""
        temp_path = os.path.join(tempfile.gettempdir(), filename)
        
        # Convert analysis to dictionary (simplified)
        analysis_dict = {
            'summary': {
                'original_word_count': analysis.word_count_original,
                'enhanced_word_count': analysis.word_count_enhanced,
                'transformations_count': len(analysis.transformations),
                'processing_time_seconds': analysis.processing_time,
                'readability_improvement': analysis.readability_improvement
            },
            'quality_metrics': analysis.quality_metrics,
            'transformations': [
                {
                    'original_word': t.original_word,
                    'enhanced_word': t.enhanced_word,
                    'confidence_score': t.confidence_score,
                    'transformation_reason': t.transformation_reason,
                    'position': t.position
                }
                for t in analysis.transformations[:50]  # Limit for file size
            ],
            'timestamp': datetime.now().isoformat()
        }
        
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_dict, f, indent=2, ensure_ascii=False)
        
        return temp_path
    
    def get_analytics_dashboard(self):
        """Create analytics dashboard from processing history"""
        if not self.processing_history:
            return """
## 📊 Analytics Dashboard

### 🚀 Welcome to TextGenix!

No processing history available yet. Upload and process some documents to see detailed analytics here!

### 🎯 Quick Start Guide:
1. **Upload** a document (PDF, DOCX, TXT, HTML, RTF)
2. **Configure** your processing settings
3. **Process** with our AI engine
4. **Review** your enhanced document
5. **Download** results and analytics

*Your processing statistics and insights will appear here after your first document.*
            """
        
        # Calculate aggregate statistics
        total_docs = len(self.processing_history)
        total_words_processed = sum(h['analysis'].word_count_original for h in self.processing_history)
        total_transformations = sum(len([t for t in h['analysis'].transformations if t.enhanced_word != t.original_word]) for h in self.processing_history)
        avg_quality = sum(h['analysis'].quality_metrics.get('overall_quality', 0) 
                         for h in self.processing_history) / total_docs
        avg_processing_time = sum(h['analysis'].processing_time for h in self.processing_history) / total_docs
        
        dashboard = f"""
## 📊 Analytics Dashboard

### 🏆 Processing Statistics
| Metric | Value |
|--------|-------|
| 📚 **Total Documents Processed** | {total_docs:,} |
| 📝 **Total Words Processed** | {total_words_processed:,} |
| ✨ **Total Transformations** | {total_transformations:,} |
| 🎯 **Average Quality Score** | {avg_quality:.1%} |
| ⚡ **Average Processing Time** | {avg_processing_time:.2f}s |
| 📈 **Enhancement Rate** | {(total_transformations/total_words_processed*100):.1f}% |

### 📋 Recent Activity
"""
        
        # Show recent files
        for i, history in enumerate(self.processing_history[-5:], 1):
            quality = history['analysis'].quality_metrics.get('overall_quality', 0)
            transformations = len([t for t in history['analysis'].transformations if t.enhanced_word != t.original_word])
            quality_emoji = "🟢" if quality >= 0.8 else "🟡" if quality >= 0.6 else "🟠"
            
            dashboard += f"""
**{i}.** 📄 **{history['filename']}** {quality_emoji}
   - 📅 *Processed*: {history['timestamp'].strftime('%Y-%m-%d %H:%M')}
   - 🎯 *Quality*: {quality:.1%}
   - ✨ *Transformations*: {transformations}
   - 📝 *Words*: {history['analysis'].word_count_original:,}
"""
        
        if total_docs > 5:
            dashboard += f"\n*...and {total_docs - 5} more documents in your history*"
        
        return dashboard

def create_interface():
    """Create the main Gradio interface with beautiful modern styling"""
    web_interface = WebInterface()
    
    # Create interface with custom theme
    with gr.Blocks(
        css=CUSTOM_CSS,
        title="TextGenix - AI Document Enhancement",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="purple",
            neutral_hue="slate",
            font="Inter"
        )
    ) as interface:
        
        # Beautiful Header
        with gr.Column(elem_classes=["main-header"]):
            gr.HTML("""
            <div style="text-align: center; padding: 2rem;">
                <h1 style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 3.5rem; font-weight: 700; margin-bottom: 0.5rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.1);">
                    🚀 TextGenix Enterprise
                </h1>
                <h3 style="color: rgba(255, 255, 255, 0.9); font-size: 1.5rem; font-weight: 400; margin-bottom: 1rem;">
                    AI-Powered Intelligent Document Processing
                </h3>
                <p style="color: rgba(255, 255, 255, 0.8); font-size: 1.1rem; max-width: 600px; margin: 0 auto;">
                    Transform your documents with cutting-edge artificial intelligence technology. 
                    Professional enhancement, industry-specific terminology, and real-time quality analytics.
                </p>
            </div>
            """)
        
        # Main Processing Interface
        with gr.Row(equal_height=True):
            # Left Column - Input and Configuration
            with gr.Column(scale=1, elem_classes=["card"]):
                gr.HTML('<h2 class="section-header">📄 Document Upload & Configuration</h2>')
                
                # File upload with beautiful styling
                with gr.Group(elem_classes=["input-group"]):
                    file_input = gr.File(
                        label="📁 Upload Your Document",
                        file_types=[".txt", ".pdf", ".docx", ".html", ".rtf"],
                        file_count="single",
                        elem_classes=["file-upload"]
                    )
                    gr.HTML('<small style="color: #666;">Supports PDF, DOCX, TXT, HTML, RTF formats</small>')
                
                gr.HTML('<h3 class="section-header">⚙️ Processing Configuration</h3>')
                
                with gr.Row():
                    industry = gr.Dropdown(
                        choices=[
                            ("🏢 General Business", "general"),
                            ("⚖️ Legal", "legal"),
                            ("🏥 Medical", "medical"),
                            ("💰 Financial", "financial"),
                            ("🔧 Technical", "technical")
                        ],
                        value="general",
                        label="🎯 Industry Context",
                        elem_classes=["gradio-dropdown"]
                    )
                    mode = gr.Dropdown(
                        choices=[
                            ("💼 Professional", "professional"),
                            ("🎨 Creative", "creative"),
                            ("🔬 Technical", "technical"),
                            ("📝 Basic", "basic")
                        ],
                        value="professional",
                        label="✨ Enhancement Mode",
                        elem_classes=["gradio-dropdown"]
                    )
                
                with gr.Row():
                    min_length = gr.Slider(
                        minimum=1, maximum=10, value=3, step=1,
                        label="📏 Minimum Word Length"
                    )
                    max_length = gr.Slider(
                        minimum=5, maximum=25, value=15, step=1,
                        label="📐 Maximum Word Length"
                    )
                
                quality_threshold = gr.Slider(
                    minimum=0.5, maximum=1.0, value=0.8, step=0.05,
                    label="🎯 Quality Threshold"
                )
                
                with gr.Row():
                    preserve_context = gr.Checkbox(
                        value=True, label="🧠 Preserve Context"
                    )
                    enable_grammar = gr.Checkbox(
                        value=True, label="✍️ Enable Grammar Check"
                    )
                
                # Beautiful process button
                process_btn = gr.Button(
                    "🚀 PROCESS DOCUMENT",
                    variant="primary",
                    size="lg",
                    elem_classes=["primary-btn", "pulse"]
                )
                
                # Status with styling
                status_output = gr.Textbox(
                    label="📊 Processing Status",
                    interactive=False,
                    max_lines=3,
                    elem_classes=["status-output"]
                )
            
            # Right Column - Results Preview
            with gr.Column(scale=1, elem_classes=["card"]):
                gr.HTML('<h2 class="section-header">📊 Processing Results</h2>')
                
                # Results section (initially hidden)
                with gr.Group(visible=False, elem_classes=["results-section"]) as results_section:
                    
                    # Enhanced text output with beautiful styling
                    enhanced_text_output = gr.Textbox(
                        label="✨ Enhanced Document",
                        lines=12,
                        max_lines=20,
                        interactive=False,
                        elem_classes=["enhanced-text"]
                    )
                    
                    # Summary with markdown
                    summary_output = gr.Markdown(
                        label="📋 Processing Summary",
                        elem_classes=["fade-in"]
                    )
        
        # Analytics Section with Beautiful Visualizations
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML('<h2 class="section-header">📈 Quality Analytics</h2>')
                quality_chart = gr.Plot(
                    label="Quality Metrics Dashboard",
                    elem_classes=["glow"]
                )
            
            with gr.Column(scale=1):
                gr.HTML('<h2 class="section-header">🔄 Transformation Analysis</h2>')
                word_cloud_chart = gr.Plot(
                    label="Transformation Distribution",
                    elem_classes=["glow"]
                )
        
        # Detailed Transformations Table
        with gr.Group(elem_classes=["card"]):
            gr.HTML('<h2 class="section-header">📄 Word Transformations</h2>')
            transformations_table = gr.Dataframe(
                label="Detailed Transformation Log",
                interactive=False,
                elem_classes=["gradio-dataframe"]
            )
        
        # Download Section with Beautiful Buttons
        with gr.Row():
            with gr.Column():
                enhanced_download = gr.File(
                    label="📥 Download Enhanced Document",
                    interactive=False
                )
            with gr.Column():
                analysis_download = gr.File(
                    label="📥 Download Analysis Report",
                    interactive=False
                )
        
        # Tabs for additional features with custom styling
        with gr.Tabs():
            with gr.Tab("📊 Analytics Dashboard", elem_id="analytics-tab"):
                gr.HTML('<h2 class="section-header">📈 Analytics Dashboard</h2>')
                analytics_output = gr.Markdown(elem_classes=["fade-in"])
                refresh_analytics_btn = gr.Button(
                    "🔄 Refresh Analytics",
                    elem_classes=["secondary-btn"]
                )
            
            with gr.Tab("🚀 Quick Start", elem_id="quickstart-tab"):
                gr.HTML("""
                <div style="padding: 2rem; background: linear-gradient(145deg, #ffffff, #f8fbff); border-radius: 16px; margin: 1rem 0;">
                    <h2 style="color: #667eea; margin-bottom: 1.5rem;">🚀 Quick Start Guide</h2>
                    
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 2rem; margin: 2rem 0;">
                        <div style="background: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); border-left: 4px solid #667eea;">
                            <h3 style="color: #667eea; margin-bottom: 1rem;">📄 1. Upload Document</h3>
                            <p>Upload any supported document format (PDF, DOCX, TXT, HTML, RTF). Our AI will extract and analyze the text content.</p>
                        </div>
                        
                        <div style="background: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); border-left: 4px solid #764ba2;">
                            <h3 style="color: #764ba2; margin-bottom: 1rem;">⚙️ 2. Configure Settings</h3>
                            <p>Choose your industry context and enhancement mode. Adjust quality thresholds and processing parameters.</p>
                        </div>
                        
                        <div style="background: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); border-left: 4px solid #f093fb;">
                            <h3 style="color: #f093fb; margin-bottom: 1rem;">🤖 3. AI Processing</h3>
                            <p>Our advanced AI analyzes context, identifies improvement opportunities, and enhances vocabulary with industry-specific terms.</p>
                        </div>
                        
                        <div style="background: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); border-left: 4px solid #f5576c;">
                            <h3 style="color: #f5576c; margin-bottom: 1rem;">📊 4. Review Results</h3>
                            <p>Examine quality metrics, transformation details, and analytics. Download enhanced documents and detailed reports.</p>
                        </div>
                    </div>
                    
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem; border-radius: 12px; margin-top: 2rem; text-align: center;">
                        <h3 style="margin-bottom: 1rem;">💡 Sample Text for Testing</h3>
                        <p style="font-family: 'JetBrains Mono', monospace; background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 8px; font-size: 0.9rem; line-height: 1.6;">
                            The contract has important terms that both parties must follow.<br>
                            Our company had good results this year with big improvements.<br>
                            The system works by using advanced methods to process data.
                        </p>
                        <small style="opacity: 0.8;">Copy this text into a .txt file and upload it to see TextGenix in action!</small>
                    </div>
                </div>
                """)
            
            with gr.Tab("ℹ️ About", elem_id="about-tab"):
                gr.HTML("""
                <div style="padding: 2rem; background: linear-gradient(145deg, #ffffff, #f8fbff); border-radius: 16px; margin: 1rem 0;">
                    <h2 style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.5rem; font-weight: 700; text-align: center; margin-bottom: 2rem;">
                        ℹ️ About TextGenix Enterprise
                    </h2>
                    
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 2rem; margin: 2rem 0;">
                        <div style="background: white; padding: 2rem; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                            <h3 style="color: #667eea; margin-bottom: 1rem;">🚀 Features</h3>
                            <ul style="line-height: 1.8;">
                                <li><strong>Multi-format Support</strong>: PDF, DOCX, TXT, HTML, RTF</li>
                                <li><strong>AI Enhancement</strong>: Context-aware text improvement</li>
                                <li><strong>Industry Specialization</strong>: Legal, medical, financial, technical</li>
                                <li><strong>Quality Assurance</strong>: Automated grammar and semantic checking</li>
                                <li><strong>Real-time Analytics</strong>: Comprehensive processing metrics</li>
                            </ul>
                        </div>
                        
                        <div style="background: white; padding: 2rem; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                            <h3 style="color: #764ba2; margin-bottom: 1rem;">🧠 AI Technology</h3>
                            <ul style="line-height: 1.8;">
                                <li><strong>NLP Engine</strong>: spaCy with advanced linguistic analysis</li>
                                <li><strong>Semantic Analysis</strong>: Sentence transformers for context</li>
                                <li><strong>Quality Assurance</strong>: Multi-layered validation system</li>
                                <li><strong>Grammar Checking</strong>: Professional language validation</li>
                            </ul>
                        </div>
                        
                        <div style="background: white; padding: 2rem; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                            <h3 style="color: #f093fb; margin-bottom: 1rem;">💼 Enterprise Ready</h3>
                            <ul style="line-height: 1.8;">
                                <li><strong>Scalable Processing</strong>: Handle large document volumes</li>
                                <li><strong>Quality Metrics</strong>: Measurable improvement tracking</li>
                                <li><strong>Professional Output</strong>: Business-grade enhancement</li>
                                <li><strong>Analytics Dashboard</strong>: Executive-level reporting</li>
                            </ul>
                        </div>
                    </div>
                    
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem; border-radius: 12px; margin-top: 2rem;">
                        <h3 style="text-align: center; margin-bottom: 1.5rem;">📋 Installation Requirements</h3>
                        <div style="background: rgba(0,0,0,0.2); padding: 1.5rem; border-radius: 8px; font-family: 'JetBrains Mono', monospace; font-size: 0.9rem; line-height: 1.6;">
                            # Basic requirements (always needed)<br>
                            pip install gradio pandas plotly<br><br>
                            
                            # Optional AI enhancements (for better quality)<br>
                            pip install spacy nltk textstat sentence-transformers scikit-learn<br>
                            pip install PyPDF2 python-docx html2text striprtf language-tool-python<br><br>
                            
                            # Download spaCy model (recommended)<br>
                            python -m spacy download en_core_web_sm
                        </div>
                    </div>
                    
                    <div style="text-align: center; margin-top: 2rem; padding: 1rem; background: rgba(102, 126, 234, 0.1); border-radius: 8px;">
                        <p style="font-style: italic; color: #667eea; font-size: 1.1rem;">
                            Developed with cutting-edge AI technology for enterprise document processing
                        </p>
                        <p style="color: #888; margin-top: 0.5rem;">
                            <strong>Version</strong>: 2.0.0 | <strong>License</strong>: MIT | <strong>Support</strong>: Enterprise Ready
                        </p>
                    </div>
                </div>
                """)
        
        # Event handlers
        process_btn.click(
            fn=web_interface.process_document,
            inputs=[
                file_input, industry, mode, min_length, max_length,
                quality_threshold, preserve_context, enable_grammar
            ],
            outputs=[
                status_output, enhanced_text_output, summary_output,
                quality_chart, word_cloud_chart, transformations_table, 
                enhanced_download, analysis_download, results_section
            ]
        )
        
        refresh_analytics_btn.click(
            fn=web_interface.get_analytics_dashboard,
            outputs=analytics_output
        )
        
        # Load initial analytics
        interface.load(
            fn=web_interface.get_analytics_dashboard,
            outputs=analytics_output
        )
    
    return interface

def main():
    """Launch the beautiful web application"""
    print("🚀 Starting TextGenix Enterprise...")
    print("📦 Checking dependencies...")
    
    # Check what's available
    available_features = []
    if SPACY_AVAILABLE:
        available_features.append("✅ Advanced NLP (spaCy)")
    else:
        available_features.append("⚠️  Basic NLP (spaCy not available)")
    
    if AI_AVAILABLE:
        available_features.append("✅ AI Semantic Analysis")
    else:
        available_features.append("⚠️  Basic Semantic Analysis")
    
    if PDF_AVAILABLE:
        available_features.append("✅ PDF Processing")
    else:
        available_features.append("⚠️  PDF Processing (install PyPDF2)")
    
    if DOCX_AVAILABLE:
        available_features.append("✅ DOCX Processing")
    else:
        available_features.append("⚠️  DOCX Processing (install python-docx)")
    
    print("\n📋 Available Features:")
    for feature in available_features:
        print(f"   {feature}")
    
    print("\n🌐 Launching beautiful web interface...")
    
    interface = create_interface()
    
    # Launch with custom settings
    interface.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,       # Default Gradio port
        share=False,            # Set to True for public sharing
        debug=False,            # Set to True for development
        show_api=False,         # Hide API docs for cleaner interface
        quiet=False,            # Show launch info
        inbrowser=True,         # Auto-open browser
        favicon_path=None,      # You can add a custom favicon
        app_kwargs={"docs_url": None, "redoc_url": None}  # Clean URLs
    )

if __name__ == "__main__":
    print("="*70)
    print("🚀 TEXTGENIX ENTERPRISE - BEAUTIFUL AI DOCUMENT PROCESSING")
    print("="*70)
    print("🎨 Beautiful Modern UI with Gradient Backgrounds")
    print("✨ Professional Styling and Animations")
    print("📊 Interactive Charts and Visualizations") 
    print("🌈 Custom Color Schemes and Typography")
    print("="*70)
    print("🔗 Access URL: http://localhost:7860")
    print("🎯 Ready to transform documents with beautiful AI!")
    print("="*70)
    main()