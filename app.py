"""
FastAPI Application for AI Contract Risk Analyzer
Complete pre-loading approach: All models loaded at startup
Direct synchronous flow: Upload → Analyze → Return Results + PDF
"""
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
import os
from datetime import datetime
from pathlib import Path
import sys
import tempfile
import io

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

# Import all services
from config.settings import settings
from config.risk_rules import ContractType
from model_manager.model_loader import ModelLoader
from utils.document_reader import DocumentReader
from utils.validators import ContractValidator
from utils.text_processor import TextProcessor
from utils.logger import ContractAnalyzerLogger, log_info, log_error

from services.contract_classifier import ContractClassifier
from services.clause_extractor import ClauseExtractor
from services.risk_analyzer import MultiFactorRiskAnalyzer
from services.term_analyzer import TermAnalyzer
from services.protection_checker import ProtectionChecker
from services.llm_interpreter import LLMClauseInterpreter
from services.negotiation_engine import NegotiationEngine
from services.market_comparator import MarketComparator

# Import PDF generator
from reporter.pdf_generator import generate_pdf_report

# Initialize logger
ContractAnalyzerLogger.setup(log_dir="logs", app_name="contract_analyzer")
logger = ContractAnalyzerLogger.get_logger()

# ============================================================================
# PYDANTIC SCHEMAS
# ============================================================================

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    timestamp: str
    models_loaded: int
    services_loaded: int
    memory_usage_mb: float

class AnalysisOptions(BaseModel):
    """Analysis options"""
    max_clauses: int = Field(default=15, ge=5, le=30)
    interpret_clauses: bool = Field(default=True)
    generate_negotiation_points: bool = Field(default=True)
    compare_to_market: bool = Field(default=True)

class AnalysisResult(BaseModel):
    """Complete analysis result"""
    analysis_id: str
    timestamp: str
    classification: Dict[str, Any]
    clauses: List[Dict[str, Any]]
    risk_analysis: Dict[str, Any]
    unfavorable_terms: List[Dict[str, Any]]
    missing_protections: List[Dict[str, Any]]
    clause_interpretations: Optional[List[Dict[str, Any]]] = None
    negotiation_points: Optional[List[Dict[str, Any]]] = None
    market_comparisons: Optional[List[Dict[str, Any]]] = None
    executive_summary: str
    metadata: Dict[str, Any]
    pdf_available: bool = True

class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: str
    timestamp: str

# ============================================================================
# SERVICE INITIALIZATION WITH FULL PRE-LOADING
# ============================================================================

class PreloadedAnalysisService:
    """Analysis service with complete pre-loading of all models"""
    
    def __init__(self):
        self.model_loader = ModelLoader()
        self.services = {}
        self.service_status = {}
        self.memory_usage_mb = 0
        self._preload_all_services()
    
    def _preload_all_services(self):
        """Pre-load ALL services and models at initialization"""
        log_info("PRE-LOADING ALL AI MODELS AND SERVICES")
        
        try:
            # Track memory usage
            initial_memory = self._get_memory_usage()
            
            # 1. Pre-load core classifier
            log_info("🔄 Pre-loading Contract Classifier...")
            self.services["classifier"] = ContractClassifier(self.model_loader)
            self.service_status["classifier"] = "loaded"
            log_info("✅ Contract Classifier loaded")
            
            # 2. Pre-load Term Analyzer
            log_info("🔄 Pre-loading Term Analyzer...")
            self.services["term_analyzer"] = TermAnalyzer()
            self.service_status["term_analyzer"] = "loaded"
            log_info("✅ Term Analyzer loaded")
            
            # 3. Pre-load Protection Checker
            log_info("🔄 Pre-loading Protection Checker...")
            self.services["protection_checker"] = ProtectionChecker()
            self.service_status["protection_checker"] = "loaded"
            log_info("✅ Protection Checker loaded")
            
            # 4. Pre-load Market Comparator
            log_info("🔄 Pre-loading Market Comparator...")
            self.services["market_comparator"] = MarketComparator(self.model_loader)
            self.service_status["market_comparator"] = "loaded"
            log_info("✅ Market Comparator loaded")
            
            # 5. Pre-load Clause Extractors for all major contract types
            log_info("🔄 Pre-loading Clause Extractors...")
            self.services["extractors"] = {}
            major_categories = ["employment", "consulting", "nda", "software", "service", "partnership"]
            
            for category in major_categories:
                try:
                    self.services["extractors"][category] = ClauseExtractor(
                        self.model_loader, contract_category=category
                    )
                    log_info(f"  ✅ Clause Extractor for {category} loaded")
                except Exception as e:
                    log_error(f"Failed to load extractor for {category}: {e}")
                    self.services["extractors"][category] = None
            
            self.service_status["extractors"] = f"loaded for {len(major_categories)} categories"
            log_info("✅ All Clause Extractors loaded")
            
            # 6. Pre-load Risk Analyzers for all contract types
            log_info("🔄 Pre-loading Risk Analyzers...")
            self.services["risk_analyzers"] = {}
            contract_types = [
                ContractType.EMPLOYMENT, ContractType.CONSULTING, ContractType.NDA,
                ContractType.SOFTWARE, ContractType.SERVICE, ContractType.PARTNERSHIP,
                ContractType.LEASE, ContractType.PURCHASE, ContractType.GENERAL
            ]
            
            for contract_type in contract_types:
                try:
                    self.services["risk_analyzers"][contract_type.value] = MultiFactorRiskAnalyzer(
                        contract_type=contract_type
                    )
                    log_info(f"  ✅ Risk Analyzer for {contract_type.value} loaded")
                except Exception as e:
                    log_error(f"Failed to load risk analyzer for {contract_type.value}: {e}")
                    self.services["risk_analyzers"][contract_type.value] = None
            
            self.service_status["risk_analyzers"] = f"loaded for {len(contract_types)} types"
            log_info("✅ All Risk Analyzers loaded")
            
            # 7. Pre-load LLM Interpreter (if available)
            log_info("🔄 Pre-loading LLM Interpreter...")
            try:
                self.services["interpreter"] = LLMClauseInterpreter()
                self.service_status["interpreter"] = "loaded"
                log_info("✅ LLM Interpreter loaded")
            except Exception as e:
                self.services["interpreter"] = None
                self.service_status["interpreter"] = f"failed: {str(e)}"
                log_info("⚠️  LLM Interpreter not available (will skip interpretation)")
            
            # 8. Pre-load Negotiation Engine (if available)
            log_info("🔄 Pre-loading Negotiation Engine...")
            try:
                self.services["negotiation_engine"] = NegotiationEngine()
                self.service_status["negotiation_engine"] = "loaded"
                log_info("✅ Negotiation Engine loaded")
            except Exception as e:
                self.services["negotiation_engine"] = None
                self.service_status["negotiation_engine"] = f"failed: {str(e)}"
                log_info("⚠️  Negotiation Engine not available (will skip negotiation points)")
            
            # Calculate memory usage
            final_memory = self._get_memory_usage()
            self.memory_usage_mb = final_memory - initial_memory
            
            log_info("🎉 ALL SERVICES PRE-LOADED SUCCESSFULLY!")
            log_info(f"📊 Memory Usage: {self.memory_usage_mb:.2f} MB")
            log_info(f"🔧 Services Loaded: {len(self.service_status)}")
            
        except Exception as e:
            log_error(f"CRITICAL: Failed to pre-load services: {e}")
            raise
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get detailed service status"""
        model_stats = self.model_loader.get_registry_stats()
        return {
            "services": self.service_status,
            "models": model_stats,
            "memory_usage_mb": self.memory_usage_mb,
            "total_services_loaded": len([s for s in self.service_status.values() if "loaded" in str(s)]),
            "total_models_loaded": model_stats.get("loaded_models", 0)
        }
    
    def analyze_contract(self, contract_text: str, options: AnalysisOptions) -> Dict[str, Any]:
        """Synchronous contract analysis using pre-loaded services"""
        try:
            log_info("Starting contract analysis with pre-loaded services...")
            
            # Step 1: Classify contract
            classification = self.services["classifier"].classify_contract(contract_text)
            classification_dict = classification.to_dict()
            actual_category = classification.category
            
            log_info(f"Contract classified as: {actual_category}")
            
            # Step 2: Get appropriate extractor
            extractor = self.services["extractors"].get(actual_category)
            if not extractor:
                # Fallback to first available extractor or create new one
                available_categories = [cat for cat, ext in self.services["extractors"].items() if ext is not None]
                if available_categories:
                    fallback_category = available_categories[0]
                    extractor = self.services["extractors"][fallback_category]
                    log_info(f"Using fallback extractor for: {fallback_category}")
                else:
                    # Create new extractor for this category
                    extractor = ClauseExtractor(self.model_loader, contract_category=actual_category)
                    self.services["extractors"][actual_category] = extractor
            
            # Extract clauses
            clauses = extractor.extract_clauses(contract_text, options.max_clauses)
            clauses_dict = [clause.to_dict() for clause in clauses]
            log_info(f"Extracted {len(clauses)} clauses")
            
            # Step 3: Map to ContractType and get appropriate risk analyzer
            contract_type_mapping = {
                'employment': ContractType.EMPLOYMENT,
                'consulting': ContractType.CONSULTING,
                'nda': ContractType.NDA,
                'technology': ContractType.SOFTWARE,
                'software': ContractType.SOFTWARE,
                'service_agreement': ContractType.SERVICE,
                'business': ContractType.PARTNERSHIP,
                'real_estate': ContractType.LEASE,
                'sales': ContractType.PURCHASE,
            }
            contract_type = contract_type_mapping.get(actual_category, ContractType.GENERAL)
            
            risk_analyzer = self.services["risk_analyzers"].get(contract_type.value)
            if not risk_analyzer:
                # Fallback to general analyzer
                risk_analyzer = self.services["risk_analyzers"]["general"]
            
            # Analyze risk
            risk_score = risk_analyzer.analyze_risk(contract_text, clauses)
            risk_dict = risk_score.to_dict()
            log_info(f"Risk analysis completed: {risk_dict['overall_score']}/100")
            
            # Step 4: Find unfavorable terms
            unfavorable_terms = self.services["term_analyzer"].analyze_unfavorable_terms(contract_text, clauses)
            unfavorable_dict = [term.to_dict() for term in unfavorable_terms]
            log_info(f"Found {len(unfavorable_terms)} unfavorable terms")
            
            # Step 5: Check missing protections
            missing_protections = self.services["protection_checker"].check_missing_protections(contract_text, clauses)
            missing_dict = [prot.to_dict() for prot in missing_protections]
            log_info(f"Found {len(missing_protections)} missing protections")
            
            # Optional steps
            interpretations_dict = None
            negotiation_dict = None
            market_dict = None
            
            if options.interpret_clauses and self.services["interpreter"]:
                try:
                    interpretations = self.services["interpreter"].interpret_clauses(
                        clauses, min(10, options.max_clauses)
                    )
                    interpretations_dict = [interp.to_dict() for interp in interpretations]
                    log_info(f"Interpreted {len(interpretations)} clauses")
                except Exception as e:
                    log_error(f"Clause interpretation failed: {e}")
                    interpretations_dict = []
            
            if options.generate_negotiation_points and self.services["negotiation_engine"]:
                try:
                    negotiation_points = self.services["negotiation_engine"].generate_negotiation_points(
                        risk_score, unfavorable_terms, missing_protections, clauses, 7
                    )
                    negotiation_dict = [point.to_dict() for point in negotiation_points]
                    log_info(f"Generated {len(negotiation_points)} negotiation points")
                except Exception as e:
                    log_error(f"Negotiation points generation failed: {e}")
                    negotiation_dict = []
            
            if options.compare_to_market:
                try:
                    market_comparisons = self.services["market_comparator"].compare_to_market(clauses)
                    market_dict = [comp.to_dict() for comp in market_comparisons]
                    log_info(f"Compared {len(market_comparisons)} clauses to market")
                except Exception as e:
                    log_error(f"Market comparison failed: {e}")
                    market_dict = []
            
            # Generate executive summary
            executive_summary = self._generate_executive_summary(
                classification_dict, risk_dict, unfavorable_dict, missing_dict
            )
            
            # Build result
            result = {
                "analysis_id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "classification": classification_dict,
                "clauses": clauses_dict,
                "risk_analysis": risk_dict,
                "unfavorable_terms": unfavorable_dict,
                "missing_protections": missing_dict,
                "clause_interpretations": interpretations_dict,
                "negotiation_points": negotiation_dict,
                "market_comparisons": market_dict,
                "executive_summary": executive_summary,
                "metadata": {
                    "text_length": len(contract_text),
                    "word_count": len(contract_text.split()),
                    "num_clauses": len(clauses),
                    "contract_type": contract_type.value,
                    "actual_category": actual_category,
                    "options": options.dict()
                },
                "pdf_available": True
            }
            
            log_info("Contract analysis completed successfully")
            return result
            
        except Exception as e:
            log_error(f"Contract analysis failed: {e}")
            raise
    
    def _generate_executive_summary(self, classification: Dict, risk_score: Dict, 
                                   unfavorable_terms: List, missing_protections: List) -> str:
        """Generate executive summary"""
        category = classification.get("category", "Unknown")
        score = risk_score.get("overall_score", 0)
        risk_level = risk_score.get("risk_level", "UNKNOWN")
        
        critical_terms = sum(1 for t in unfavorable_terms if t.get('severity') == 'critical')
        critical_protections = sum(1 for p in missing_protections if p.get('importance') == 'critical')
        
        if score >= 80:
            risk_msg = "CRITICAL ATTENTION REQUIRED"
        elif score >= 60:
            risk_msg = "SIGNIFICANT CONCERNS"
        elif score >= 40:
            risk_msg = "MODERATE RISK"
        else:
            risk_msg = "LOW RISK"
        
        return f"This {category} contract scored {score}/100 ({risk_level.upper()} risk). {risk_msg}. Found {len(unfavorable_terms)} unfavorable terms ({critical_terms} critical) and {len(missing_protections)} missing protections ({critical_protections} critical). Review detailed analysis below."

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI-powered contract risk analysis with complete model pre-loading",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS
)

# Initialize pre-loaded analysis service
analysis_service = PreloadedAnalysisService()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def validate_file(file: UploadFile) -> tuple[bool, str]:
    """File validation using settings from config"""
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in settings.ALLOWED_EXTENSIONS:
        return False, f"Invalid file type. Allowed: {', '.join(settings.ALLOWED_EXTENSIONS)}"
    
    file.file.seek(0, 2)
    size = file.file.tell()
    file.file.seek(0)
    
    if size > settings.MAX_UPLOAD_SIZE:
        return False, f"File too large. Max size: {settings.MAX_UPLOAD_SIZE / (1024*1024)}MB"
    
    if size == 0:
        return False, "File is empty"
    
    return True, "OK"

def read_contract_file(file: UploadFile) -> str:
    """Read contract text from file using DocumentReader"""
    file_ext = os.path.splitext(file.filename)[1].lower()
    file_type = "pdf" if file_ext == ".pdf" else "docx" if file_ext == ".docx" else "txt"
    
    reader = DocumentReader()
    file_contents = reader.read_file(file.file, file_type)
    
    # Handle both string and dict return types from DocumentReader
    if isinstance(file_contents, dict):
        return file_contents.get('text', '') or file_contents.get('content', '')
    else:
        return str(file_contents)

def validate_contract_text(text: str) -> tuple[bool, str]:
    """Validate contract text using settings"""
    if not text or not text.strip():
        return False, "Contract text is empty"
    
    if len(text) < settings.MIN_CONTRACT_LENGTH:
        return False, f"Contract text too short. Minimum {settings.MIN_CONTRACT_LENGTH} characters required."
    
    if len(text) > settings.MAX_CONTRACT_LENGTH:
        return False, f"Contract text too long. Maximum {settings.MAX_CONTRACT_LENGTH} characters allowed."
    
    return True, "OK"

# ============================================================================
# API ROUTES
# ============================================================================

@app.get("/")
async def serve_frontend():
    """Serve the frontend"""
    return FileResponse("static/index.html")

@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with service status"""
    service_status = analysis_service.get_service_status()
    
    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        timestamp=datetime.now().isoformat(),
        models_loaded=service_status["total_models_loaded"],
        services_loaded=service_status["total_services_loaded"],
        memory_usage_mb=service_status["memory_usage_mb"]
    )

@app.get("/api/v1/status")
async def get_detailed_status():
    """Get detailed service status"""
    return analysis_service.get_service_status()

@app.post("/api/v1/analyze/file", response_model=AnalysisResult)
async def analyze_contract_file(
    file: UploadFile = File(...),
    max_clauses: int = Form(15),
    interpret_clauses: bool = Form(True),
    generate_negotiation_points: bool = Form(True),
    compare_to_market: bool = Form(True)
):
    """Analyze uploaded contract file - DIRECT SYNC FLOW"""
    try:
        # Validate file
        is_valid, message = validate_file(file)
        if not is_valid:
            raise HTTPException(status_code=400, detail=message)
        
        # Read contract text
        contract_text = read_contract_file(file)
        
        # Validate contract text
        is_valid_text, text_message = validate_contract_text(contract_text)
        if not is_valid_text:
            raise HTTPException(status_code=400, detail=text_message)
        
        # Validate contract structure using ContractValidator
        validator = ContractValidator()
        is_valid_contract, contract_type, confidence = validator.is_valid_contract(contract_text)
        
        if not is_valid_contract:
            raise HTTPException(status_code=400, detail=f"Invalid contract: {confidence}")
        
        # Create analysis options
        options = AnalysisOptions(
            max_clauses=min(max_clauses, settings.MAX_CLAUSES_TO_ANALYZE),
            interpret_clauses=interpret_clauses,
            generate_negotiation_points=generate_negotiation_points,
            compare_to_market=compare_to_market
        )
        
        # Perform analysis (SYNCHRONOUS with pre-loaded services)
        result = analysis_service.analyze_contract(contract_text, options)
        
        log_info(f"File analysis completed", 
                filename=file.filename,
                analysis_id=result["analysis_id"],
                risk_score=result["risk_analysis"]["overall_score"])
        
        return AnalysisResult(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        log_error(f"File analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/api/v1/analyze/text", response_model=AnalysisResult)
async def analyze_contract_text(
    contract_text: str = Form(..., description="Contract text to analyze"),
    max_clauses: int = Form(15),
    interpret_clauses: bool = Form(True),
    generate_negotiation_points: bool = Form(True),
    compare_to_market: bool = Form(True)
):
    """Analyze pasted contract text - DIRECT SYNC FLOW"""
    try:
        # Validate contract text
        is_valid, message = validate_contract_text(contract_text)
        if not is_valid:
            raise HTTPException(status_code=400, detail=message)
        
        # Validate contract structure using ContractValidator
        validator = ContractValidator()
        is_valid_contract, contract_type, confidence = validator.is_valid_contract(contract_text)
        
        if not is_valid_contract:
            raise HTTPException(status_code=400, detail=f"Invalid contract: {confidence}")
        
        # Create analysis options
        options = AnalysisOptions(
            max_clauses=min(max_clauses, settings.MAX_CLAUSES_TO_ANALYZE),
            interpret_clauses=interpret_clauses,
            generate_negotiation_points=generate_negotiation_points,
            compare_to_market=compare_to_market
        )
        
        # Perform analysis (SYNCHRONOUS with pre-loaded services)
        result = analysis_service.analyze_contract(contract_text, options)
        
        log_info(f"Text analysis completed", 
                analysis_id=result["analysis_id"],
                risk_score=result["risk_analysis"]["overall_score"])
        
        return AnalysisResult(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        log_error(f"Text analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/api/v1/generate-pdf")
async def generate_pdf_from_analysis(analysis_result: Dict[str, Any]):
    """Generate PDF from analysis results"""
    try:
        pdf_buffer = generate_pdf_report(analysis_result)
        
        analysis_id = analysis_result.get('analysis_id', 'report')
        return Response(
            content=pdf_buffer.getvalue(),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=contract_analysis_{analysis_id}.pdf"
            }
        )
    except Exception as e:
        log_error(f"PDF generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF: {str(e)}")

@app.get("/api/v1/categories")
async def get_contract_categories():
    """Get list of supported contract categories"""
    try:
        categories = analysis_service.services["classifier"].get_all_categories()
        return {"categories": categories}
    except Exception as e:
        log_error(f"Categories fetch failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get categories: {str(e)}")

@app.post("/api/v1/validate/file")
async def validate_contract_file(file: UploadFile = File(...)):
    """Quick validation endpoint"""
    try:
        is_valid, message = validate_file(file)
        if not is_valid:
            return {"valid": False, "message": message}
        
        contract_text = read_contract_file(file)
        
        # Validate text length
        is_valid_text, text_message = validate_contract_text(contract_text)
        if not is_valid_text:
            return {"valid": False, "message": text_message}
        
        # Validate contract structure using ContractValidator
        validator = ContractValidator()
        report = validator.get_validation_report(contract_text)
        
        return {
            "valid": report["scores"]["total"] > 50 and is_valid_text,
            "message": "Contract appears valid" if report["scores"]["total"] > 50 else "May not be a valid contract",
            "confidence": report["scores"]["total"],
            "report": report
        }
        
    except Exception as e:
        log_error(f"File validation failed: {e}")
        raise HTTPException(status_code=400, detail=f"Validation failed: {str(e)}")

@app.post("/api/v1/validate/text")
async def validate_contract_text_endpoint(contract_text: str = Form(...)):
    """Validate pasted contract text"""
    try:
        # Validate text length
        is_valid, message = validate_contract_text(contract_text)
        if not is_valid:
            return {"valid": False, "message": message}
        
        # Validate contract structure using ContractValidator
        validator = ContractValidator()
        report = validator.get_validation_report(contract_text)
        
        return {
            "valid": report["scores"]["total"] > 50 and is_valid,
            "message": "Contract appears valid" if report["scores"]["total"] > 50 else "May not be a valid contract",
            "confidence": report["scores"]["total"],
            "report": report
        }
        
    except Exception as e:
        log_error(f"Text validation failed: {e}")
        raise HTTPException(status_code=400, detail=f"Validation failed: {str(e)}")

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=str(exc.detail),
            timestamp=datetime.now().isoformat()
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    log_error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            timestamp=datetime.now().isoformat()
        ).dict()
    )

# ============================================================================
# STARTUP & SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Startup event - Services are already pre-loaded"""
    log_info(f"🚀 {settings.APP_NAME} v{settings.APP_VERSION} STARTED")
    log_info(f"📍 Server: {settings.HOST}:{settings.PORT}")
    log_info(f"🔧 All models and services pre-loaded")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event"""
    log_info("🛑 Shutting down server...")
    log_info("✅ Server shutdown complete")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        workers=1,  # Single worker for synchronous flow
        log_level=settings.LOG_LEVEL.lower()
    )