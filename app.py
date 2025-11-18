# app.py
# DEPENDENCIES
import os
import sys
import time
import json
import uuid
import signal
import uvicorn
import numpy as np
from typing import Any
from typing import List
from typing import Dict
from pathlib import Path
from fastapi import File
from fastapi import Form
from pydantic import Field
from fastapi import FastAPI
from fastapi import Request
from typing import Optional
from datetime import datetime
from pydantic import BaseModel
from fastapi import UploadFile
from fastapi import HTTPException
from fastapi.responses import Response
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from utils.logger import log_info
from utils.logger import log_error
from config.settings import settings
from config.risk_rules import ContractType
from services.data_models import RiskScore
from utils.validators import ContractValidator
from utils.text_processor import TextProcessor
from services.data_models import SummaryContext 
from utils.logger import ContractAnalyzerLogger
from services.risk_analyzer import RiskAnalyzer
from services.term_analyzer import TermAnalyzer
from services.data_models import ExtractedClause
from services.data_models import UnfavorableTerm
from utils.document_reader import DocumentReader
from model_manager.llm_manager import LLMManager
from services.data_models import NegotiationPoint
from services.data_models import ContractCategory
from model_manager.llm_manager import LLMProvider
from model_manager.model_loader import ModelLoader
from services.data_models import MissingProtection
from services.data_models import RiskInterpretation
from services.data_models import NegotiationPlaybook
from reporter.pdf_generator import PDFReportGenerator
from services.data_models import ClauseInterpretation
from reporter.pdf_generator import generate_pdf_report
from services.summary_generator import SummaryGenerator
from services.clause_extractor import RiskClauseExtractor
from services.negotiation_engine import NegotiationEngine
from services.llm_interpreter import LLMClauseInterpreter
from services.protection_checker import ProtectionChecker
from services.contract_classifier import ContractClassifier
from services.clause_extractor import ComprehensiveClauseExtractor


# ============================================================================
# CUSTOM SERIALIZATION METHODS
# ============================================================================
class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        
        elif isinstance(obj, (np.int32, np.int64, np.int8, np.uint8)):
            return int(obj)
        
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        
        elif isinstance(obj, np.bool_):
            return bool(obj)
        
        elif hasattr(obj, 'item'):
            return obj.item()
        
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        
        elif hasattr(obj, 'dict'):
            return obj.dict()
        
        elif isinstance(obj, (set, tuple)):
            return list(obj)
        
        return super().default(obj)


class NumpyJSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        
        return json.dumps(obj          = content,
                          ensure_ascii = False,
                          allow_nan    = False,
                          indent       = None,
                          separators   = (",", ":"),
                          cls          = NumpyJSONEncoder,
                         ).encode("utf-8")


def convert_numpy_types(obj: Any) -> Any:
    if obj is None:
        return None
    
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    
    elif isinstance(obj, (list, tuple, set)):
        return [convert_numpy_types(item) for item in obj]
    
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    
    elif isinstance(obj, (np.int32, np.int64, np.int8, np.uint8)):
        return int(obj)
    
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    
    elif isinstance(obj, np.bool_):
        return bool(obj)
    
    elif hasattr(obj, 'item'):
        return obj.item()
    
    elif hasattr(obj, 'to_dict'):
        return convert_numpy_types(obj.to_dict())
    
    elif hasattr(obj, 'dict'):
        return convert_numpy_types(obj.dict())
    
    else:
        return obj


def safe_serialize_response(data: Any) -> Any:
    return convert_numpy_types(data)


# PYDANTIC SCHEMAS 
class SerializableBaseModel(BaseModel):
    def dict(self, *args, **kwargs) -> Dict[str, Any]:
        data = super().dict(*args, **kwargs)
        return convert_numpy_types(data)

    
    def json(self, *args, **kwargs) -> str:
        data = self.dict(*args, **kwargs)
        return json.dumps(data, cls = NumpyJSONEncoder, *args, **kwargs)


class HealthResponse(SerializableBaseModel):
    status          : str
    version         : str
    timestamp       : str
    models_loaded   : int
    services_loaded : int
    memory_usage_mb : float


class AnalysisOptions(SerializableBaseModel):
    max_clauses                 : int  = Field(default = 100, ge = 5, le = 50)
    interpret_clauses           : bool = Field(default = True)
    generate_negotiation_points : bool = Field(default = True)
    compare_to_market           : bool = Field(default = False)  # Disabled for now


class AnalysisResult(SerializableBaseModel):
    analysis_id            : str
    timestamp              : str
    classification         : Dict[str, Any]
    clauses                : List[Dict[str, Any]]
    risk_analysis          : Dict[str, Any]
    unfavorable_terms      : List[Dict[str, Any]]
    missing_protections    : List[Dict[str, Any]]
    clause_interpretations : Optional[List[Dict[str, Any]]] = None
    negotiation_points     : Optional[List[Dict[str, Any]]] = None
    market_comparisons     : Optional[List[Dict[str, Any]]] = None
    executive_summary      : str
    metadata               : Dict[str, Any]
    pdf_available          : bool                           = True


class ErrorResponse(SerializableBaseModel):
    error     : str
    detail    : str
    timestamp : str


class FileValidationResponse(SerializableBaseModel):
    valid      : bool
    message    : str
    confidence : Optional[float]          = None
    report     : Optional[Dict[str, Any]] = None


# SERVICE INITIALIZATION WITH FULL PIPELINE INTEGRATION
class PreloadedAnalysisService:
    """
    Analysis service with complete pipeline integration
    """
    def __init__(self):
        self.model_loader    = ModelLoader()
        self.llm_manager     = LLMManager() 
        self.services        = dict()
        self.service_status  = dict()
        self.memory_usage_mb = 0

        self._preload_all_services()


    def _preload_all_services(self):
        """
        Pre-load ALL services and models at initialization
        """
        log_info("PRE-LOADING ALL AI MODELS AND SERVICES")
        try:
            initial_memory = self._get_memory_usage()

            # Pre-load Contract Classifier
            log_info("🔄 Pre-loading Contract Classifier...")
            try:
                self.services["classifier"]       = ContractClassifier(self.model_loader)
                self.service_status["classifier"] = "loaded"
                log_info("✅ Contract Classifier loaded")
            
            except Exception as e:
                log_error(f"Failed to load ContractClassifier: {repr(e)}")
                raise

            # Pre-load ComprehensiveClauseExtractor as base for RiskClauseExtractor
            log_info("🔄 Pre-loading Comprehensive Clause Extractor...")
            try:
                self.services["comprehensive_extractor"]       = ComprehensiveClauseExtractor(self.model_loader)
                self.service_status["comprehensive_extractor"] = "loaded"

                log_info("✅ Comprehensive Clause Extractor loaded")
            
            except Exception as e:
                log_error(f"Failed to load ComprehensiveClauseExtractor: {repr(e)}")
                raise

            # Initialize RiskClauseExtractor with default type (will be recreated per analysis)
            log_info("🔄 Initializing Risk-Focused Clause Extractor...")
            try:
                self.services["clause_extractor"]       = RiskClauseExtractor(model_loader  = self.model_loader,
                                                                              contract_type = ContractType.GENERAL,
                                                                             )
                self.service_status["clause_extractor"] = "loaded"

                log_info("✅ Risk-Focused Clause Extractor initialized")

            except Exception as e:
                log_error(f"Failed to initialize RiskClauseExtractor: {repr(e)}")
                raise

            # Pre-load RiskAnalyzer
            log_info("🔄 Pre-loading Risk Analyzer...")
            try:
                # RiskAnalyzer orchestrates other services but doesn't need to initialize them separately
                self.services["risk_analyzer"]       = RiskAnalyzer(self.model_loader)
                self.service_status["risk_analyzer"] = "loaded"

                log_info("✅ Comprehensive Risk Analyzer loaded")
            
            except Exception as e:
                log_error(f"Failed to load RiskAnalyzer: {repr(e)}")
                raise

            # Pre-load LLM Interpreter 
            log_info("🔄 Pre-loading LLM Interpreter...")
            try:
                self.services["llm_interpreter"]       = LLMClauseInterpreter(self.llm_manager)
                self.service_status["llm_interpreter"] = "loaded"

                log_info("✅ LLM Interpreter loaded")

            except Exception as e:
                self.services["llm_interpreter"]       = None
                self.service_status["llm_interpreter"] = f"failed: {repr(e)}"

                log_info("⚠️  LLM Interpreter not available")

            # Pre-load Negotiation Engine
            log_info("🔄 Pre-loading Negotiation Engine...")
            try:
                # Initialize with LLM manager - ensure constructor args match
                self.services["negotiation_engine"]       = NegotiationEngine(llm_manager      = self.llm_manager, 
                                                                              default_provider = LLMProvider.OLLAMA,
                                                                             )
                self.service_status["negotiation_engine"] = "loaded"

                log_info("✅ Negotiation Engine loaded")
            
            except Exception as e:
                self.services["negotiation_engine"]       = None
                self.service_status["negotiation_engine"] = f"failed: {repr(e)}"

                log_info("⚠️  Negotiation Engine not available")

            # Pre-load Summary Generator
            log_info("🔄 Pre-loading Summary Generator...")
            try:
                # Initialize with LLM manager
                self.services["summary_generator"]       = SummaryGenerator(llm_manager = self.llm_manager)
                self.service_status["summary_generator"] = "loaded"
                
                log_info("✅ Summary Generator loaded")
            
            except Exception as e:
                # Fallback if initialization fails
                self.services["summary_generator"]       = SummaryGenerator()
                self.service_status["summary_generator"] = "fallback_loaded"
                
                log_info("⚠️  Summary Generator using fallback mode")

            # Pre-load Unfavorable Term Analyzer
            log_info("🔄 Pre-loading Unfavorable Term Analyzer...")
            try:
                # Initialize with default contract type, will be updated per analysis
                self.services["term_analyzer"]       = TermAnalyzer(contract_type = ContractType.GENERAL)
                self.service_status["term_analyzer"] = "loaded"
                
                log_info("✅ Unfavorable Term Analyzer loaded")
            
            except Exception as e:
                log_error(f"Failed to load TermAnalyzer: {repr(e)}")
                raise

            # Pre-load Missing Protection Checker
            log_info("🔄 Pre-loading Missing Protection Checker...")
            try:
                # Initialize with default contract type, will be updated per analysis
                self.services["protection_checker"]       = ProtectionChecker(contract_type = ContractType.GENERAL)
                self.service_status["protection_checker"] = "loaded"
                
                log_info("✅ Protection Checker loaded")
            
            except Exception as e:
                log_error(f"Failed to load ProtectionChecker: {repr(e)}")
                raise

            # Calculate memory usage
            final_memory         = self._get_memory_usage()
            self.memory_usage_mb = final_memory - initial_memory
            
            log_info("🎉 ALL SERVICES PRE-LOADED SUCCESSFULLY!")
            log_info(f"📊 Memory Usage: {self.memory_usage_mb:.2f} MB")
            log_info(f"🔧 Services Loaded: {len(self.service_status)}")

        except Exception as e:
            log_error(f"CRITICAL: Failed to pre-load services: {e}")
            raise


    def _get_memory_usage(self) -> float:
        """
        Get current memory usage in MB
        """
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
       
        except ImportError:
            return 0.0


    def _create_fallback_negotiation_points(self, risk_score, unfavorable_terms, missing_protections):
        """
        Create basic negotiation points when engine fails
        """
        fallback_points = list()
        # Add top unfavorable terms
        for term in unfavorable_terms[:5]:
            fallback_points.append({"priority"             : 1 if term.severity == "critical" else 2,
                                    "category"             : term.category,
                                    "issue"                : term.term,
                                    "current_language"     : "See contract clause",
                                    "proposed_language"    : term.suggested_fix or "Request balanced language",
                                    "rationale"            : term.explanation,
                                    "estimated_difficulty" : "medium"
                                  })
        # Add critical missing protections
        for protection in [p for p in missing_protections if (p.importance == "critical")][:5]:
            fallback_points.append({"priority"             : 1,
                                    "category"             : protection.categories[0] if protection.categories else "general",
                                    "issue"                : f"Add {protection.protection}",
                                    "current_language"     : "[MISSING]",
                                    "proposed_language"    : protection.suggested_language or protection.recommendation,
                                    "rationale"            : protection.explanation,
                                    "estimated_difficulty" : "medium"
                                  })
        return fallback_points


    def get_service_status(self) -> Dict[str, Any]:
        """
        Get detailed service status
        """
        model_stats = self.model_loader.get_registry_stats()
        return {"services"              : self.service_status,
                "models"                : model_stats,
                "memory_usage_mb"       : self.memory_usage_mb,
                "total_services_loaded" : len([s for s in self.service_status.values() if "loaded" in str(s)]),
                "total_models_loaded"   : model_stats.get("loaded_models", 0),
               }


    def analyze_contract(self, contract_text: str, options: AnalysisOptions) -> Dict[str, Any]:
        """
        Complete contract analysis using full pipeline
        """
        try:
            log_info("Starting comprehensive contract analysis pipeline...")

            # Classify contract
            classification                    = self.services["classifier"].classify_contract(contract_text)
            classification_dict               = safe_serialize_response(classification.to_dict())
            log_info(f"Contract classified as: {classification.category}")
            
            # Debug logging for classification
            log_info(f"Classification details - Confidence: {classification.confidence:.3f}, "
                     f"Subcategory: {classification.subcategory}, "
                     f"Keywords found: {len(classification.detected_keywords)}",
                    )

            # Get ContractType enum for downstream services
            contract_type_enum                = self._get_contract_type_enum(category_str = classification.category)

            # Re-initialize RiskClauseExtractor with correct contract type: crucial for category mapping in risk analysis
            if (hasattr(self.services["clause_extractor"], 'contract_type')):
                self.services["clause_extractor"].contract_type    = contract_type_enum
                self.services["clause_extractor"].category_weights = self.services["clause_extractor"].risk_rules.get_adjusted_weights(contract_type_enum)
                
                log_info(f"Updated RiskClauseExtractor for contract type: {contract_type_enum.value}")
            
            else:
                # Fallback: create new instance if update not possible
                self.services["clause_extractor"] = RiskClauseExtractor(model_loader  = self.model_loader,
                                                                        contract_type = contract_type_enum,
                                                                       )
                log_info(f"Re-initialized RiskClauseExtractor for contract type: {contract_type_enum.value}")

            # Extract Risk Focused clauses (outputs risk categories)
            clauses      = list()
            clauses_dict = list()
            try:
                # Try risk-focused extraction first
                clauses = self.services["clause_extractor"].extract_risk_clauses(contract_text = contract_text,
                                                                                 max_clauses   = options.max_clauses,
                                                                                )
                
                log_info(f"Extracted {len(clauses)} risk-focused clauses")
            
            except Exception as e:
                log_error(f"Risk-focused clause extraction failed: {repr(e)}")
                # Fallback to comprehensive extraction
                try:
                    log_info("Attempting fallback to comprehensive clause extraction...")
                    clauses = self.services["comprehensive_extractor"].extract_clauses(contract_text = contract_text,
                                                                                       max_clauses   = options.max_clauses,
                                                                                      )
                    
                    log_info(f"Fallback extracted {len(clauses)} comprehensive clauses")
                
                except Exception as fallback_error:
                    log_error(f"Comprehensive clause extraction also failed: {repr(fallback_error)}")
                    clauses = []

            # Process clauses regardless of extraction method
            if clauses:
                clauses_dict      = [safe_serialize_response(clause.to_dict()) for clause in clauses]
                # Debug logging for clause extraction
                clause_categories = [clause.category for clause in clauses]
                unique_categories = list(set(clause_categories))
                
                log_info(f"Clause categories extracted: {unique_categories}")
                
                # Log risk scores if available
                risk_scores = [getattr(clause, 'risk_score', 0) for clause in clauses if hasattr(clause, 'risk_score')]
                
                if risk_scores:
                    avg_risk = sum(risk_scores) / len(risk_scores)
                    log_info(f"Average clause risk score: {avg_risk:.2f}")

            # Analyze UNFAVORABLE TERMS (outputs risk categories)
            unfavorable_terms      = list()
            unfavorable_terms_dict = list()
            
            try:
                # Update term analyzer with correct contract type
                if hasattr(self.services["term_analyzer"], 'contract_type'):
                    self.services["term_analyzer"].contract_type    = contract_type_enum
                    self.services["term_analyzer"].category_weights = self.services["term_analyzer"].risk_rules.get_adjusted_weights(contract_type_enum)
                    
                    log_info(f"Updated TermAnalyzer for contract type: {contract_type_enum.value}")
                
                unfavorable_terms      = self.services["term_analyzer"].analyze_unfavorable_terms(contract_text = contract_text,
                                                                                                  clauses       = clauses,
                                                                                                  contract_type = contract_type_enum,
                                                                                                 )
                
                unfavorable_terms_dict = [safe_serialize_response(term.to_dict()) for term in unfavorable_terms]
                
                log_info(f"Analyzed {len(unfavorable_terms)} unfavorable terms")
                
                # Debug logging for term analysis
                if unfavorable_terms:
                    severity_counts = dict()
                    for term in unfavorable_terms:
                        severity_counts[term.severity] = severity_counts.get(term.severity, 0) + 1
                    
                    log_info(f"Term severity distribution: {severity_counts}")
                    
                    # Log top 10 highest risk terms
                    top_terms = sorted(unfavorable_terms, key = lambda x: x.risk_score, reverse = True)[:10]
                    for i, term in enumerate(top_terms):
                        log_info(f"Top term {i+1}: {term.term} (Risk: {term.risk_score}, Severity: {term.severity})")
            
            except Exception as e:
                log_error(f"Unfavorable terms analysis failed: {repr(e)}")
                
                # Continue with empty terms but log the error
                unfavorable_terms      = list()
                unfavorable_terms_dict = list()

            # Check for Missing Protections (outputs risk categories)
            missing_protections      = list()
            missing_protections_dict = list()
            
            try:
                # Update protection checker with correct contract type
                if hasattr(self.services["protection_checker"], 'contract_type'):
                    self.services["protection_checker"].contract_type         = contract_type_enum
                    self.services["protection_checker"].protection_priorities = self.services["protection_checker"]._get_contract_type_priorities()
                    
                    log_info(f"Updated ProtectionChecker for contract type: {contract_type_enum.value}")
                
                missing_protections      = self.services["protection_checker"].check_missing_protections(contract_text = contract_text,
                                                                                                         clauses       = clauses,
                                                                                                         contract_type = contract_type_enum,
                                                                                                        )
                missing_protections_dict = [safe_serialize_response(prot.to_dict()) for prot in missing_protections]
                
                log_info(f"Checked for {len(missing_protections)} missing protections")
                
                # Debug logging for protection analysis
                if missing_protections:
                    importance_counts = dict()
                    for prot in missing_protections:
                        importance_counts[prot.importance] = importance_counts.get(prot.importance, 0) + 1
                    
                    log_info(f"Missing protection importance: {importance_counts}")
                    
                    # Log top 10 highest risk missing protections
                    top_protections = sorted(missing_protections, key = lambda x: x.risk_score, reverse = True)[:10]
                    
                    for i, prot in enumerate(top_protections):
                        log_info(f"Top missing protection {i+1}: {prot.protection} (Risk: {prot.risk_score}, Importance: {prot.importance})")
            
            except Exception as e:
                log_error(f"Missing protection analysis failed: {repr(e)}")
                
                # Continue with empty protections but log the error
                missing_protections      = list()
                missing_protections_dict = list()

            # Perform Complete Risk Analysis
            risk_score                        = self.services["risk_analyzer"].analyze_contract_risk(contract_text = contract_text)
            risk_dict                         = safe_serialize_response(risk_score.to_dict())
            log_info(f"Risk analysis completed: {risk_score.overall_score}/100")

            # Generate LLM Interpretations (if available)
            risk_interpretation = None
            
            if self.services["llm_interpreter"]:
                try:
                    risk_interpretation = self.services["llm_interpreter"].interpret_with_risk_context(clauses             = clauses,
                                                                                                       unfavorable_terms   = unfavorable_terms,
                                                                                                       missing_protections = missing_protections,
                                                                                                       contract_type       = contract_type_enum,
                                                                                                       overall_risk_score  = risk_score.overall_score,
                                                                                                       max_clauses         = len(clauses), 
                                                                                                       provider            = LLMProvider.OLLAMA,
                                                                                                      )
                    log_info("LLM risk interpretation generated")
                
                except Exception as e:
                    log_error(f"LLM interpretation failed: {repr(e)}")
                    # Continue without LLM interpretation
            
            else:
                # If LLM is not available, create a basic interpretation object to pass downstream
                risk_interpretation = RiskInterpretation(overall_risk_explanation = f"Contract risk score: {risk_score.overall_score}/100 ({risk_score.risk_level}).",
                                                         key_concerns             = [f"Risk level: {risk_score.risk_level}"],
                                                         negotiation_strategy     = "Address critical terms identified in analysis.",
                                                         market_comparison        = "Compare with industry standards.",
                                                         clause_interpretations   = [],
                                                        )


            # Generate Negotiation Playbook (uses full context)
            negotiation_playbook = None
            negotiation_dict     = list()

            if self.services["negotiation_engine"]:
                try:
                    # Ensure we have proper objects, not dicts
                    unfavorable_terms_objects   = unfavorable_terms  
                    missing_protections_objects = missing_protections  
                    
                    # Create a fallback risk interpretation if LLM failed (already handled above)
                    negotiation_playbook        = self.services["negotiation_engine"].generate_comprehensive_playbook(risk_analysis       = risk_score,
                                                                                                                      risk_interpretation = risk_interpretation,
                                                                                                                      unfavorable_terms   = unfavorable_terms_objects,
                                                                                                                      missing_protections = missing_protections_objects,
                                                                                                                      clauses             = clauses, 
                                                                                                                      contract_type       = contract_type_enum,
                                                                                                                      max_points          = len(clauses),
                                                                                                                     )

                    negotiation_dict     = [safe_serialize_response(point.to_dict()) for point in negotiation_playbook.critical_points]

                    log_info(f"Negotiation playbook generated with {len(negotiation_playbook.critical_points)} points")

                except Exception as e:
                    log_error(f"Negotiation playbook generation failed: {repr(e)}")
                    
                    # Create fallback negotiation points
                    negotiation_dict = self._create_fallback_negotiation_points(risk_score, unfavorable_terms, missing_protections)
            
            else:
                # If negotiation engine is not available, create fallback points
                negotiation_dict = self._create_fallback_negotiation_points(risk_score, unfavorable_terms, missing_protections)


            # Generate Executive Summary (uses full context)
            executive_summary = self.services["summary_generator"].generate_executive_summary(contract_text        = contract_text,
                                                                                              classification       = classification,
                                                                                              risk_analysis        = risk_score,
                                                                                              risk_interpretation  = risk_interpretation,
                                                                                              negotiation_playbook = negotiation_playbook,
                                                                                              unfavorable_terms    = unfavorable_terms, 
                                                                                              missing_protections  = missing_protections, 
                                                                                              clauses              = clauses, 
                                                                                             )
            log_info("Executive summary generated")

            # Build final result matching frontend expectations
            result                            = {"analysis_id"            : str(uuid.uuid4()),
                                                 "timestamp"              : datetime.now().isoformat(),
                                                 "classification"         : classification_dict,
                                                 "clauses"                : clauses_dict,
                                                 "risk_analysis"          : risk_dict, 
                                                 "unfavorable_terms"      : unfavorable_terms_dict,
                                                 "missing_protections"    : missing_protections_dict,
                                                 "clause_interpretations" : [safe_serialize_response(interp.to_dict()) for interp in (risk_interpretation.clause_interpretations if risk_interpretation else [])],
                                                 "negotiation_points"     : negotiation_dict,
                                                 "market_comparisons"     : [],
                                                 "executive_summary"      : executive_summary,
                                                 "metadata"               : {"text_length"               : len(contract_text),
                                                                             "word_count"                : len(contract_text.split()),
                                                                             "num_clauses"               : len(clauses),
                                                                             "contract_type"             : contract_type_enum.value,
                                                                             "actual_category"           : classification.category,
                                                                             "subcategory"               : classification.subcategory,
                                                                             "classification_confidence" : classification.confidence,
                                                                             "detected_keywords"         : classification.detected_keywords,
                                                                             "options"                   : options.dict(),
                                                                            },
                                                 "pdf_available"          : True,
                                                }

            log_info("Contract analysis completed successfully")
            return result

        except Exception as e:
            log_error(f"Contract analysis failed: {repr(e)}")
            raise


    def _score_to_risk_level(self, score: float) -> str:
        """
        Convert risk score to risk level string
        """
        if (score >= 80):
            return "Critical"
        
        elif (score >= 60):
            return "High"
        
        elif (score >= 40):
            return "Medium"
        
        else:
            return "Low"


    def _get_contract_type_enum(self, category_str: str) -> ContractType:
        """
        Convert category string to ContractType enum with fallback
        """
        mapping       = {'employment'  : ContractType.EMPLOYMENT,
                         'consulting'  : ContractType.CONSULTING,
                         'nda'         : ContractType.NDA,
                         'software'    : ContractType.SOFTWARE,
                         'service'     : ContractType.SERVICE,
                         'partnership' : ContractType.PARTNERSHIP,
                         'lease'       : ContractType.LEASE,
                         'purchase'    : ContractType.PURCHASE,
                         'general'     : ContractType.GENERAL,
                        }

        contract_type = mapping.get(category_str, ContractType.GENERAL)
        
        log_info(f"Mapping category '{category_str}' to ContractType: {contract_type.value}")
        
        return contract_type



# FASTAPI APPLICATION : Global instances
analysis_service : Optional[PreloadedAnalysisService] = None
app_start_time                                        = time.time()

# Initialize logger
ContractAnalyzerLogger.setup(log_dir  = "logs", 
                             app_name = "contract_analyzer",
                            )

logger = ContractAnalyzerLogger.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global analysis_service
    log_info(f"🚀 {settings.APP_NAME} v{settings.APP_VERSION} STARTING UP...")
    log_info("=" * 80)
    
    try:
        analysis_service = PreloadedAnalysisService()
        log_info("✅ All services initialized successfully")

    except Exception as e:
        log_error(f"Startup failed: {e}")
        raise

    log_info(f"📍 Server: {settings.HOST}:{settings.PORT}")
    log_info("=" * 80)
    log_info("✅ AI Contract Risk Analyzer Ready!")

    try:
        yield

    finally:
        log_info("🛑 Shutting down server...")
        log_info("✅ Server shutdown complete")

# Define the application
app        = FastAPI(title                 = settings.APP_NAME,
                    version                = settings.APP_VERSION,
                    description            = "AI-powered contract risk analysis",
                    docs_url               = "/api/docs",
                    redoc_url              = "/api/redoc",
                    default_response_class = NumpyJSONResponse,
                    lifespan               = lifespan,
                   )

# Get absolute paths
BASE_DIR   = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"

# Serve static files
app.mount("/static", StaticFiles(directory = str(STATIC_DIR)), name = "static")

# CORS middleware
app.add_middleware(CORSMiddleware,
                   allow_origins     = settings.CORS_ORIGINS,
                   allow_credentials = settings.CORS_ALLOW_CREDENTIALS,
                   allow_methods     = settings.CORS_ALLOW_METHODS,
                   allow_headers     = settings.CORS_ALLOW_HEADERS,
                  )


# HELPER FUNCTIONS
def validate_file(file: UploadFile) -> tuple[bool, str]:
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    if file_extension not in settings.ALLOWED_EXTENSIONS:
        return False, f"Invalid file type. Allowed: {', '.join(settings.ALLOWED_EXTENSIONS)}"
    
    file.file.seek(0, 2)
    size = file.file.tell()
    
    file.file.seek(0)
    
    if (size > settings.MAX_UPLOAD_SIZE):
        return False, f"File too large. Max size: {settings.MAX_UPLOAD_SIZE / (1024*1024):.1f}MB"
    
    if (size == 0):
        return False, "File is empty"
    
    return True, "OK"


def read_contract_file(file) -> str:
    """
    Read contract file and return text content.
    """
    reader         = DocumentReader()
    # Extract file extension without dot
    filename       = file.filename.lower()
    file_extension = Path(filename).suffix.lower().lstrip('.')
    
    # If no extension found, try to detect from content or default to pdf
    if not file_extension:
        file_extension = "pdf"
        print(f"📁 DEBUG app.py - No extension found, defaulting to: '{file_extension}'")
    
    file_contents = reader.read_file(file.file, file_extension)
    if (not file_contents or not file_contents.strip()):
        raise ValueError("Could not extract text from file")
    
    return file_contents


def validate_contract_text(text: str) -> tuple[bool, str]:
    if not text or not text.strip():
        return False, "Contract text is empty"
    
    if (len(text) < settings.MIN_CONTRACT_LENGTH):
        return False, f"Contract text too short. Minimum {settings.MIN_CONTRACT_LENGTH} characters required."
    
    if (len(text) > settings.MAX_CONTRACT_LENGTH):
        return False, f"Contract text too long. Maximum {settings.MAX_CONTRACT_LENGTH} characters allowed."
    
    return True, "OK"



# API ROUTES
@app.get("/")
async def serve_frontend():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/v1/health", response_model = HealthResponse)
async def health_check():
    if not analysis_service:
        raise HTTPException(status_code = 503, 
                            detail      = "Service not initialized",
                           )

    service_status = analysis_service.get_service_status()

    return HealthResponse(status          = "healthy",
                          version         = settings.APP_VERSION,
                          timestamp       = datetime.now().isoformat(),
                          models_loaded   = service_status["total_models_loaded"],
                          services_loaded = service_status["total_services_loaded"],
                          memory_usage_mb = service_status["memory_usage_mb"],
                         )


@app.get("/api/v1/status")
async def get_detailed_status():
    if not analysis_service:
        raise HTTPException(status_code = 503, 
                            detail      = "Service not initialized",
                           )

    return analysis_service.get_service_status()


@app.post("/api/v1/analyze/file", response_model = AnalysisResult)
async def analyze_contract_file(file: UploadFile = File(...), max_clauses: int = Form(100), interpret_clauses: bool = Form(True),   
                                generate_negotiation_points: bool = Form(True), compare_to_market: bool = Form(False)):
    if not analysis_service:
        raise HTTPException(status_code = 503, 
                            detail      = "Service not initialized",
                           )

    try:
        # Validate file
        is_valid, message = validate_file(file)

        if not is_valid:
            raise HTTPException(status_code = 400, 
                                detail      = message,
                               )

        # Read contract text
        contract_text               = read_contract_file(file)

        # Validate contract text
        is_valid_text, text_message = validate_contract_text(contract_text)

        if not is_valid_text:
            raise HTTPException(status_code = 400, 
                                detail      = text_message,
                               )

        # Validate contract structure using ContractValidator
        validator                                    = ContractValidator()
        is_valid_contract, contract_type, confidence = validator.is_valid_contract(contract_text)

        if not is_valid_contract:
            raise HTTPException(status_code = 400, 
                                detail      = f"Invalid contract: {confidence}",
                               )

        # Create analysis options
        options                                      = AnalysisOptions(max_clauses                 = max_clauses,
                                                                       interpret_clauses           = interpret_clauses,
                                                                       generate_negotiation_points = generate_negotiation_points,
                                                                       compare_to_market           = compare_to_market,
                                                                      )
        # Perform analysis
        result                                       = analysis_service.analyze_contract(contract_text, options)

        log_info(f"File analysis completed", 
                 filename    = file.filename,
                 analysis_id = result["analysis_id"],
                 risk_score  = result["risk_analysis"]["overall_score"],
                )

        return AnalysisResult(**result)

    except HTTPException:
        raise

    except Exception as e:
        log_error(f"File analysis failed: {repr(e)}")

        raise HTTPException(status_code = 500, 
                            detail      = f"Analysis failed: {repr(e)}",
                           )


@app.post("/api/v1/analyze/text", response_model = AnalysisResult)
async def analyze_contract_text(contract_text: str = Form(..., description="Contract text to analyze"), max_clauses: int = Form(100), interpret_clauses: bool = Form(True),
                                generate_negotiation_points: bool = Form(True), compare_to_market: bool = Form(False)):
    if not analysis_service:
        raise HTTPException(status_code = 503, 
                            detail      = "Service not initialized",
                           )
    try:
        # Validate contract text length first
        is_valid, message = validate_contract_text(contract_text)
        
        if not is_valid:
            raise HTTPException(status_code = 400, 
                                detail      = message,
                               )

        # Validate contract structure using ContractValidator
        validator                                   = ContractValidator()
        is_valid_contract, validation_type, message = validator.is_valid_contract(contract_text)
        
        if not is_valid_contract:
            error_message = message if "does not appear to be a legal contract" in message else "The provided document does not appear to be a legal contract. Please upload a valid contract for analysis."
            raise HTTPException(status_code = 400, 
                                detail      = error_message,
                               )

        # Create analysis options
        options = AnalysisOptions(max_clauses                 = max_clauses,
                                  interpret_clauses           = interpret_clauses,
                                  generate_negotiation_points = generate_negotiation_points,
                                  compare_to_market           = compare_to_market,
                                 )
        # Perform analysis
        result  = analysis_service.analyze_contract(contract_text, options)

        log_info(f"Text analysis completed", 
                 analysis_id = result["analysis_id"],
                 risk_score  = result["risk_analysis"]["overall_score"],
                )
        
        return AnalysisResult(**result)
    
    except HTTPException:
        raise
    
    except Exception as e:
        log_error(f"Text analysis failed: {repr(e)}")
        
        raise HTTPException(status_code = 500, 
                            detail      = f"Analysis failed: {repr(e)}",
                           )


@app.post("/api/v1/generate-pdf")
async def generate_pdf_from_analysis(analysis_result: Dict[str, Any]):
    try:
        # Pass the full analysis_result dictionary to the PDF generator
        pdf_buffer  = generate_pdf_report(analysis_result = analysis_result) 
        analysis_id = analysis_result.get('analysis_id', 'report')
        
        return Response(content    = pdf_buffer.getvalue(),
                        media_type = "application/pdf",
                        headers    = {"Content-Disposition": f"attachment; filename=contract_analysis_{analysis_id}.pdf"}
                       )

    except Exception as e:
        log_error(f"PDF generation failed: {repr(e)}")
        
        raise HTTPException(status_code = 500, 
                            detail      = f"Failed to generate PDF: {repr(e)}",
                           )


@app.get("/api/v1/categories")
async def get_contract_categories():
    if not analysis_service:
        raise HTTPException(status_code = 503, 
                            detail      = "Service not initialized",
                           )
    
    try:
        # Get categories from classifier
        categories       = analysis_service.services["classifier"].get_all_categories()
        
        # Get descriptions for each category
        category_details = list()
        
        for category in categories:
            description   = analysis_service.services["classifier"].get_category_description(category)
            subcategories = analysis_service.services["classifier"].get_subcategories(category)
            category_details.append({"name"          : category,
                                     "description"   : description,
                                     "subcategories" : subcategories,
                                   })
        
        return {"categories": category_details}
    
    except Exception as e:
        log_error(f"Categories fetch failed: {repr(e)}")
        raise HTTPException(status_code = 500, 
                            detail      = f"Failed to get categories: {repr(e)}")


@app.post("/api/v1/validate/file", response_model = FileValidationResponse)
async def validate_contract_file_endpoint(file: UploadFile = File(...)):
    try:
        is_valid, message = validate_file(file)
        if not is_valid:
            return FileValidationResponse(valid   = False,
                                          message = message,
                                         )

        contract_text = read_contract_file(file)

        # Validate text length
        is_valid_text, text_message = validate_contract_text(contract_text)
        
        if not is_valid_text:
            return FileValidationResponse(valid   = False, 
                                          message = text_message,
                                         )

        # Validate contract structure using ContractValidator
        validator = ContractValidator()
        report    = validator.get_validation_report(contract_text)
        
        return FileValidationResponse(valid      = (report["scores"]["total"] > 50) and is_valid_text,
                                      message    = "Contract appears valid" if (report["scores"]["total"] > 50) else "May not be a valid contract",
                                      confidence = report["scores"]["total"],
                                      report     = report,
                                     )
    
    except Exception as e:
        log_error(f"File validation failed: {e}")
        
        raise HTTPException(status_code = 400, 
                            detail      = f"Validation failed: {repr(e)}",
                           )


@app.post("/api/v1/validate/text", response_model = FileValidationResponse)
async def validate_contract_text_endpoint(contract_text: str = Form(...)):
    try:
        # Validate text length
        is_valid, message = validate_contract_text(contract_text)
        
        if not is_valid:
            return FileValidationResponse(valid   = False, 
                                          message = message,
                                         )

        # Validate contract structure using ContractValidator
        validator = ContractValidator()
        report    = validator.get_validation_report(contract_text)
        
        return FileValidationResponse(valid      = (report["scores"]["total"] > 50) and is_valid,
                                      message    = "Contract appears valid" if (report["scores"]["total"] > 50) else "May not be a valid contract",
                                      confidence = report["scores"]["total"],
                                      report     = report,
                                     )
    
    except Exception as e:
        log_error(f"Text validation failed: {repr(e)}")
        raise HTTPException(status_code = 400, 
                            detail      = f"Validation failed: {repr(e)}",
                           )


# ERROR HANDLERS AND MIDDLEWARE
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return NumpyJSONResponse(status_code = exc.status_code,
                             content     = ErrorResponse(error     = exc.detail,
                                                         detail    = str(exc.detail),
                                                         timestamp = datetime.now().isoformat(),
                                                        ).dict()
                            )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    log_error(f"Unhandled exception: {exc}")
    
    return NumpyJSONResponse(status_code = 500,
                             content     = ErrorResponse(error     = "Internal server error",
                                                         detail    = str(exc),
                                                         timestamp = datetime.now().isoformat(),
                                                        ).dict()
                            )


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time   = time.time()
    response     = await call_next(request)
    process_time = time.time() - start_time
    
    log_info(f"API Request: {request.method} {request.url.path} - Status: {response.status_code} - Duration: {process_time:.3f}s")
    
    return response



# MAIN 
if __name__ == "__main__":
    def signal_handler(sig, frame):
        print("\n👋 Received Ctrl+C, shutting down gracefully...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        uvicorn.run("app:app",
                    host      = settings.HOST,
                    port      = settings.PORT,
                    reload    = settings.RELOAD,
                    workers   = settings.WORKERS,
                    log_level = settings.LOG_LEVEL.lower(),
                   )

    except KeyboardInterrupt:
        print("\n🎯 Server stopped by user")

    except Exception as e:
        log_error(f"Server error: {e}")

        sys.exit(1)
