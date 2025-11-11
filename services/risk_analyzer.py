"""
Multi-Factor Risk Analyzer with sophisticated rule-based scoring
Analyzes contracts using keyword severity, structural patterns, clause-level analysis,
industry benchmarks, and missing protections detection
"""

import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.risk_rules import RiskRules, ContractType
from services.clause_extractor import ExtractedClause
from utils.logger import ContractAnalyzerLogger, log_info, log_error
from utils.text_processor import TextProcessor


@dataclass
class RiskBreakdownItem:
    """Individual risk category breakdown"""
    category: str
    score: int  # 0-100
    summary: str
    findings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "category": self.category,
            "score": self.score,
            "summary": self.summary,
            "findings": self.findings
        }


@dataclass
class RiskScore:
    """Comprehensive risk score with detailed breakdown"""
    overall_score: int  # 0-100
    risk_level: str  # "CRITICAL", "HIGH", "MEDIUM", "LOW"
    category_scores: Dict[str, int]
    risk_factors: List[str]
    detailed_findings: Dict[str, List[str]]
    benchmark_comparison: Dict[str, str]
    risk_breakdown: List[RiskBreakdownItem]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "overall_score": self.overall_score,
            "risk_level": self.risk_level,
            "category_scores": self.category_scores,
            "risk_factors": self.risk_factors,
            "detailed_findings": self.detailed_findings,
            "benchmark_comparison": self.benchmark_comparison,
            "risk_breakdown": [item.to_dict() for item in self.risk_breakdown]
        }


class MultiFactorRiskAnalyzer:
    """
    Sophisticated multi-factor risk analysis engine
    
    Analysis Factors:
    1. Keyword severity scoring (critical/high/medium keywords)
    2. Structural pattern analysis (risky contract patterns)
    3. Clause-level detailed analysis (using extracted clauses)
    4. Industry benchmark comparison
    5. Missing protections check
    6. Contract type-specific weight adjustments
    """
    
    def __init__(self, contract_type: ContractType = ContractType.GENERAL):
        """
        Initialize risk analyzer
        
        Args:
            contract_type: Type of contract for specialized analysis
        """
        self.contract_type = contract_type
        self.rules = RiskRules()
        self.adjusted_weights = self.rules.get_adjusted_weights(contract_type)
        self.text_processor = TextProcessor(use_spacy=False)
        self.logger = ContractAnalyzerLogger.get_logger()
        
        log_info("MultiFactorRiskAnalyzer initialized",
                contract_type=contract_type.value,
                adjusted_weights=self.adjusted_weights)
    
    # =========================================================================
    # MAIN ANALYSIS METHOD
    # =========================================================================
    
    @ContractAnalyzerLogger.log_execution_time("analyze_risk")
    def analyze_risk(self, contract_text: str, 
                    clauses: List[ExtractedClause]) -> RiskScore:
        """
        Comprehensive multi-factor risk analysis
        
        Args:
            contract_text: Full contract text
            clauses: Extracted clauses from ClauseExtractor
        
        Returns:
            RiskScore object with detailed analysis
        """
        
        log_info("Starting risk analysis",
                text_length=len(contract_text),
                num_clauses=len(clauses),
                contract_type=self.contract_type.value)
        
        # Initialize scoring containers
        category_scores = defaultdict(list)
        risk_factors = []
        detailed_findings = defaultdict(list)
        
        # Factor 1: Keyword Severity Scoring
        keyword_risks = self._score_keywords(contract_text)
        log_info("Keyword analysis complete", 
                critical_score=keyword_risks.get('critical', 0),
                high_score=keyword_risks.get('high', 0))
        
        # Factor 2: Structural Pattern Analysis
        pattern_risks = self._analyze_patterns(contract_text)
        log_info(f"Pattern analysis found {len(pattern_risks)} risky patterns")
        
        # Factor 3: Clause-Level Analysis
        clause_risks = self._analyze_clauses(clauses)
        log_info(f"Clause analysis complete for {len(clause_risks)} categories")
        
        # Factor 4: Missing Protections
        missing_risks = self._check_missing_protections(contract_text, clauses)
        log_info(f"Missing protections analysis complete")
        
        # Factor 5: Industry Benchmark Comparison
        benchmark_comparison = self._compare_to_benchmarks(contract_text, clauses)
        log_info(f"Benchmark comparison complete")
        
        # Aggregate scores by category
        for category in self.adjusted_weights.keys():
            category_risk = self._calculate_category_risk(
                category=category,
                keyword_risks=keyword_risks,
                pattern_risks=pattern_risks,
                clause_risks=clause_risks,
                missing_risks=missing_risks,
                benchmark_comparison=benchmark_comparison
            )
            category_scores[category] = category_risk["score"]
            detailed_findings[category] = category_risk["findings"]
            
            if category_risk["score"] >= self.rules.RISK_THRESHOLDS["high"]:
                risk_factors.append(category)
        
        # Calculate weighted overall score
        overall_score = self._calculate_weighted_score(category_scores)
        risk_level = self._get_risk_level(overall_score)
        
        # Create risk breakdown items
        risk_breakdown = self._create_risk_breakdown(
            category_scores,
            detailed_findings
        )
        
        result = RiskScore(
            overall_score=overall_score,
            risk_level=risk_level,
            category_scores=dict(category_scores),
            risk_factors=risk_factors,
            detailed_findings=dict(detailed_findings),
            benchmark_comparison=benchmark_comparison,
            risk_breakdown=risk_breakdown
        )
        
        log_info("Risk analysis complete",
                overall_score=overall_score,
                risk_level=risk_level,
                high_risk_categories=len(risk_factors))
        
        return result
    
    # =========================================================================
    # FACTOR 1: KEYWORD SEVERITY SCORING
    # =========================================================================
    
    def _score_keywords(self, text: str) -> Dict[str, int]:
        """
        Score text based on keyword severity tiers
        
        Returns:
            Dictionary with 'critical', 'high', 'medium' scores
        """
        text_lower = text.lower()
        scores = defaultdict(int)
        
        # Critical keywords (Tier 1)
        for keyword, weight in self.rules.CRITICAL_KEYWORDS.items():
            if keyword in text_lower:
                count = text_lower.count(keyword)
                scores["critical"] += weight * min(count, 3)  # Cap at 3 occurrences
        
        # High-risk keywords (Tier 2)
        for keyword, weight in self.rules.HIGH_RISK_KEYWORDS.items():
            if keyword in text_lower:
                count = text_lower.count(keyword)
                scores["high"] += weight * min(count, 2)
        
        # Medium-risk keywords (Tier 3)
        for keyword, weight in self.rules.MEDIUM_RISK_KEYWORDS.items():
            if keyword in text_lower:
                count = text_lower.count(keyword)
                scores["medium"] += weight * min(count, 2)
        
        return dict(scores)
    
    # =========================================================================
    # FACTOR 2: STRUCTURAL PATTERN ANALYSIS
    # =========================================================================
    
    def _analyze_patterns(self, text: str) -> List[Dict]:
        """
        Detect risky structural patterns in contract
        
        Returns:
            List of detected pattern dictionaries
        """
        findings = []
        
        for pattern, risk_points, description in self.rules.RISKY_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                findings.append({
                    "pattern": description,
                    "risk_points": risk_points,
                    "match": match.group(0)[:100],  # First 100 chars
                    "position": match.start()
                })
        
        return findings
    
    # =========================================================================
    # FACTOR 3: CLAUSE-LEVEL DETAILED ANALYSIS
    # =========================================================================
    
    def _analyze_clauses(self, clauses: List[ExtractedClause]) -> Dict[str, List[Dict]]:
        """
        Deep dive into each clause with specific risk factors
        
        Returns:
            Dictionary mapping categories to clause analysis results
        """
        clause_analysis = defaultdict(list)
        
        for clause in clauses:
            # Get risk factors for this clause category
            if clause.category in self.rules.CLAUSE_RISK_FACTORS:
                analysis = self._analyze_single_clause(clause)
                clause_analysis[clause.category].append(analysis)
        
        return dict(clause_analysis)
    
    def _analyze_single_clause(self, clause: ExtractedClause) -> Dict:
        """
        Analyze a single clause with detailed risk factors
        
        Returns:
            Dictionary with risk_score and findings
        """
        risk_config = self.rules.CLAUSE_RISK_FACTORS.get(clause.category, {})
        base_risk = risk_config.get("base_risk", 50)
        
        risk_score = base_risk
        findings = []
        
        text_lower = clause.text.lower()
        
        # Check red flags
        if "red_flags" in risk_config:
            for flag, adjustment in risk_config["red_flags"].items():
                if flag in text_lower:
                    risk_score += adjustment
                    severity = "increases" if adjustment > 0 else "decreases"
                    findings.append(f"Found '{flag}' ({severity} risk by {abs(adjustment)} points)")
        
        # Special handling for specific clause types
        if clause.category == "non_compete":
            duration_risk = self._analyze_noncompete_duration(clause.text)
            risk_score += duration_risk["adjustment"]
            findings.extend(duration_risk["findings"])
            
            scope_risk = self._analyze_noncompete_scope(clause.text)
            risk_score += scope_risk["adjustment"]
            findings.extend(scope_risk["findings"])
        
        elif clause.category == "termination":
            notice_risk = self._analyze_notice_period(clause.text)
            risk_score += notice_risk["adjustment"]
            findings.extend(notice_risk["findings"])
        
        elif clause.category == "indemnification":
            mutual_risk = self._analyze_indemnification_mutuality(clause.text)
            risk_score += mutual_risk["adjustment"]
            findings.extend(mutual_risk["findings"])
        
        elif clause.category == "compensation":
            clarity_risk = self._analyze_compensation_clarity(clause.text)
            risk_score += clarity_risk["adjustment"]
            findings.extend(clarity_risk["findings"])
        
        elif clause.category == "intellectual_property":
            scope_risk = self._analyze_ip_scope(clause.text)
            risk_score += scope_risk["adjustment"]
            findings.extend(scope_risk["findings"])
        
        # Cap score between 0 and 100
        risk_score = max(0, min(100, risk_score))
        
        return {
            "clause_reference": clause.reference,
            "risk_score": risk_score,
            "findings": findings,
            "confidence": clause.confidence
        }
    
    def _analyze_noncompete_duration(self, text: str) -> Dict:
        """Analyze non-compete duration reasonableness"""
        duration_pattern = r'(\d+)\s*(year|yr|month|mo)s?'
        matches = re.findall(duration_pattern, text, re.IGNORECASE)
        
        if not matches:
            return {"adjustment": 0, "findings": ["No specific duration found"]}
        
        # Convert to months
        duration_months = 0
        for num, unit in matches:
            months = int(num) * (12 if 'year' in unit.lower() or 'yr' in unit.lower() else 1)
            duration_months = max(duration_months, months)
        
        # Get benchmark
        industry = self._detect_industry()
        benchmark = self.rules.INDUSTRY_BENCHMARKS["non_compete_duration"][industry]
        
        if duration_months <= benchmark["reasonable"]:
            return {"adjustment": -10, "findings": [f"Duration of {duration_months} months is reasonable"]}
        elif duration_months <= benchmark["standard"]:
            return {"adjustment": 0, "findings": [f"Duration of {duration_months} months is standard"]}
        elif duration_months <= benchmark["excessive"]:
            return {"adjustment": +15, "findings": [f"Duration of {duration_months} months is lengthy"]}
        else:
            return {"adjustment": +30, "findings": [f"Duration of {duration_months} months is excessive"]}
    
    def _analyze_noncompete_scope(self, text: str) -> Dict:
        """Analyze non-compete scope reasonableness"""
        text_lower = text.lower()
        adjustment = 0
        findings = []
        
        scope_config = self.rules.CLAUSE_RISK_FACTORS["non-compete"]["scope_keywords"]
        
        for keyword, adj in scope_config.items():
            if keyword in text_lower:
                adjustment += adj
                severity = "reasonable" if adj < 0 else "concerning"
                findings.append(f"Scope includes '{keyword}' ({severity})")
        
        return {"adjustment": adjustment, "findings": findings}
    
    def _analyze_notice_period(self, text: str) -> Dict:
        """Analyze termination notice period balance"""
        notice_pattern = r'(\d+)\s*days?\s*(?:notice|prior\s+notice)'
        matches = re.findall(notice_pattern, text, re.IGNORECASE)
        
        if len(matches) < 2:
            return {"adjustment": 0, "findings": ["Notice period analysis inconclusive"]}
        
        periods = [int(m) for m in matches]
        
        if len(periods) >= 2:
            ratio = max(periods) / min(periods)
            
            if ratio >= 4:
                return {"adjustment": +25, "findings": [f"Notice periods highly imbalanced ({max(periods)} vs {min(periods)} days)"]}
            elif ratio >= 3:
                return {"adjustment": +18, "findings": [f"Notice periods significantly imbalanced ({max(periods)} vs {min(periods)} days)"]}
            elif ratio >= 2:
                return {"adjustment": +10, "findings": [f"Notice periods moderately imbalanced ({max(periods)} vs {min(periods)} days)"]}
            else:
                return {"adjustment": -5, "findings": [f"Notice periods balanced ({max(periods)} vs {min(periods)} days)"]}
        
        return {"adjustment": 0, "findings": ["Could not determine notice period balance"]}
    
    def _analyze_indemnification_mutuality(self, text: str) -> Dict:
        """Check if indemnification is mutual or one-sided"""
        text_lower = text.lower()
        
        mutual_indicators = ["mutual", "both parties", "each party", "reciprocal"]
        one_sided_indicators = ["employee shall indemnify", "consultant shall indemnify", 
                               "contractor shall indemnify", "you shall indemnify"]
        
        has_mutual = any(ind in text_lower for ind in mutual_indicators)
        has_one_sided = any(ind in text_lower for ind in one_sided_indicators)
        
        if has_mutual and not has_one_sided:
            return {"adjustment": -15, "findings": ["Mutual indemnification (balanced)"]}
        elif has_one_sided:
            return {"adjustment": +20, "findings": ["One-sided indemnification (unfavorable)"]}
        else:
            return {"adjustment": 0, "findings": ["Indemnification mutuality unclear"]}
    
    def _analyze_compensation_clarity(self, text: str) -> Dict:
        """Analyze clarity of compensation terms"""
        text_lower = text.lower()
        adjustment = 0
        findings = []
        
        # Check for vague terms
        vague_terms = ["to be determined", "tbd", "subject to review", "discretionary"]
        for term in vague_terms:
            if term in text_lower:
                adjustment += 10
                findings.append(f"Vague compensation term: '{term}'")
        
        # Check for specific amounts
        if re.search(r'\$[\d,]+', text):
            adjustment -= 10
            findings.append("Specific monetary amount provided (good)")
        
        return {"adjustment": adjustment, "findings": findings}
    
    def _analyze_ip_scope(self, text: str) -> Dict:
        """Analyze IP assignment scope"""
        text_lower = text.lower()
        adjustment = 0
        findings = []
        
        # Overly broad indicators
        broad_terms = ["all work product", "anything created", "whether or not related"]
        for term in broad_terms:
            if term in text_lower:
                adjustment += 15
                findings.append(f"Overly broad IP term: '{term}'")
        
        # Protective terms
        protective_terms = ["prior ip excluded", "personal projects excluded"]
        for term in protective_terms:
            if term in text_lower:
                adjustment -= 15
                findings.append(f"Protective IP term present: '{term}'")
        
        return {"adjustment": adjustment, "findings": findings}
    
    # =========================================================================
    # FACTOR 4: MISSING PROTECTIONS CHECK
    # =========================================================================
    
    def _check_missing_protections(self, text: str, 
                                   clauses: List[ExtractedClause]) -> Dict[str, int]:
        """
        Check for missing critical protections
        
        Returns:
            Dictionary mapping categories to risk scores for missing items
        """
        text_lower = text.lower()
        missing_risks = defaultdict(int)
        
        for protection_id, config in self.rules.PROTECTION_CHECKLIST.items():
            is_present = self._check_protection_present(protection_id, text_lower, clauses)
            
            if not is_present:
                risk = config["risk_if_missing"]
                for category in config["categories"]:
                    missing_risks[category] += risk
        
        return dict(missing_risks)
    
    def _check_protection_present(self, protection_id: str, 
                                 text_lower: str, 
                                 clauses: List[ExtractedClause]) -> bool:
        """Check if a specific protection is present"""
        
        protection_indicators = {
            "for_cause_definition": ["for cause", "cause defined", "grounds for termination"],
            "severance_provision": ["severance", "severance pay", "separation pay"],
            "mutual_indemnification": ["mutual indemnification", "both parties shall indemnify"],
            "liability_cap": ["liability cap", "limited to", "maximum liability"],
            "prior_ip_exclusion": ["prior intellectual property", "existing ip", "background ip"],
            "confidentiality_duration": ["confidentiality period", "for a period of"],
            "dispute_resolution": ["arbitration", "mediation", "dispute resolution"],
            "change_control_process": ["change order", "change request", "amendment process"]
        }
        
        indicators = protection_indicators.get(protection_id, [])
        return any(indicator in text_lower for indicator in indicators)
    
    # =========================================================================
    # FACTOR 5: INDUSTRY BENCHMARK COMPARISON
    # =========================================================================
    
    def _compare_to_benchmarks(self, text: str, 
                               clauses: List[ExtractedClause]) -> Dict[str, str]:
        """
        Compare contract terms to industry benchmarks
        
        Returns:
            Dictionary with human-readable comparisons
        """
        comparisons = {}
        
        # Non-compete duration
        nc_clauses = [c for c in clauses if c.category == "non_compete"]
        if nc_clauses:
            duration_months = self._extract_duration_months(nc_clauses[0].text)
            industry = self._detect_industry()
            benchmark = self.rules.INDUSTRY_BENCHMARKS["non_compete_duration"][industry]
            
            if duration_months <= benchmark["reasonable"]:
                comparisons["non_compete_duration"] = "✓ Within reasonable market standards"
            elif duration_months <= benchmark["standard"]:
                comparisons["non_compete_duration"] = "⚠ Meets market standard (consider negotiating)"
            else:
                comparisons["non_compete_duration"] = "✗ Exceeds market standards significantly"
        
        # Notice period
        term_clauses = [c for c in clauses if c.category == "termination"]
        if term_clauses:
            notice_days = self._extract_notice_period(term_clauses[0].text)
            level = "professional"  # Could be dynamic
            benchmark = self.rules.INDUSTRY_BENCHMARKS["notice_period_days"][level]
            
            if notice_days >= benchmark["reasonable"]:
                comparisons["notice_period"] = "✓ Generous notice period"
            elif notice_days >= benchmark["standard"]:
                comparisons["notice_period"] = "⚠ Standard notice period"
            else:
                comparisons["notice_period"] = "✗ Below market standard"
        
        # Liability cap
        liab_clauses = [c for c in clauses if c.category == "liability"]
        if liab_clauses:
            has_cap = "liability cap" in liab_clauses[0].text.lower() or "limited to" in liab_clauses[0].text.lower()
            if has_cap:
                comparisons["liability_cap"] = "✓ Liability cap present (good protection)"
            else:
                comparisons["liability_cap"] = "✗ No liability cap (high risk)"
        
        return comparisons
    
    def _extract_duration_months(self, text: str) -> int:
        """Extract duration in months from text"""
        duration_pattern = r'(\d+)\s*(year|yr|month|mo)s?'
        matches = re.findall(duration_pattern, text, re.IGNORECASE)
        
        if not matches:
            return 0
        
        months = 0
        for num, unit in matches:
            months = max(months, int(num) * (12 if 'year' in unit.lower() or 'yr' in unit.lower() else 1))
        
        return months
    
    def _extract_notice_period(self, text: str) -> int:
        """Extract notice period in days from text"""
        notice_pattern = r'(\d+)\s*days?\s*(?:notice|prior\s+notice)'
        matches = re.findall(notice_pattern, text, re.IGNORECASE)
        
        if not matches:
            return 0
        
        return max([int(m) for m in matches])
    
    def _detect_industry(self) -> str:
        """Detect industry from contract type"""
        mapping = {
            ContractType.EMPLOYMENT: "tech",
            ContractType.SOFTWARE: "tech",
            ContractType.CONSULTING: "tech",
            ContractType.NDA: "general"
        }
        return mapping.get(self.contract_type, "general")
    
    # =========================================================================
    # AGGREGATION & SCORING
    # =========================================================================
    
    def _calculate_category_risk(self, category: str,
                                keyword_risks: Dict[str, int],
                                pattern_risks: List[Dict],
                                clause_risks: Dict[str, List[Dict]],
                                missing_risks: Dict[str, int],
                                benchmark_comparison: Dict[str, str]) -> Dict:
        """
        Calculate risk score for a specific category
        
        Returns:
            Dictionary with 'score' and 'findings'
        """
        
        # Base score from keywords
        keyword_score = 0
        if "critical" in keyword_risks:
            keyword_score += min(keyword_risks["critical"] * 0.5, 30)
        if "high" in keyword_risks:
            keyword_score += min(keyword_risks["high"] * 0.3, 20)
        if "medium" in keyword_risks:
            keyword_score += min(keyword_risks["medium"] * 0.2, 10)
        
        # Pattern-based risk
        pattern_score = sum([p["risk_points"] for p in pattern_risks]) * 0.4
        pattern_score = min(pattern_score, 25)
        
        # Clause-level analysis
        clause_score = 0
        relevant_clauses = []
        category_mapping = self._get_category_mapping()
        
        if category in category_mapping:
            mapped_categories = category_mapping[category]
            for clause_cat in mapped_categories:
                if clause_cat in clause_risks:
                    clause_analyses = clause_risks[clause_cat]
                    if clause_analyses:
                        avg_clause_risk = sum([c["risk_score"] for c in clause_analyses]) / len(clause_analyses)
                        clause_score = max(clause_score, avg_clause_risk)
                        relevant_clauses.extend(clause_analyses)
        
        # Missing protections penalty
        missing_score = missing_risks.get(category, 0)
        
        # Combine factors (weighted average)
        if clause_score > 0:
            # If we have clause-level analysis, weight it heavily
            final_score = (
                clause_score * 0.50 +
                keyword_score * 0.20 +
                pattern_score * 0.15 +
                missing_score * 0.15
            )
        else:
            # Fallback to keyword + pattern + missing
            final_score = (
                keyword_score * 0.40 +
                pattern_score * 0.35 +
                missing_score * 0.25
            )
        
        # Cap score
        final_score = max(0, min(100, int(final_score)))
        
        # Generate findings
        findings = []
        if relevant_clauses:
            for clause in relevant_clauses[:3]:  # Top 3 clauses
                findings.extend(clause["findings"][:2])  # Top 2 findings per clause
        
        return {
            "score": final_score,
            "findings": findings[:5]  # Max 5 findings per category
        }
    
    def _get_category_mapping(self) -> Dict[str, List[str]]:
        """Map risk categories to clause categories"""
        return {
            "restrictive_covenants": ["non_compete", "confidentiality"],
            "termination_rights": ["termination"],
            "penalties_liability": ["indemnification", "liability", "warranty"],
            "compensation_benefits": ["compensation"],
            "intellectual_property": ["intellectual_property"]
        }
    
    def _calculate_weighted_score(self, category_scores: Dict[str, int]) -> int:
        """Calculate weighted average of category scores"""
        total_score = 0
        total_weight = 0
        
        for category, weight in self.adjusted_weights.items():
            if category in category_scores:
                total_score += category_scores[category] * weight
                total_weight += weight
        
        return int(total_score / total_weight) if total_weight > 0 else 50
    
    def _get_risk_level(self, score: int) -> str:
        """Get risk level from score"""
        if score >= self.rules.RISK_THRESHOLDS["critical"]:
            return "CRITICAL"
        elif score >= self.rules.RISK_THRESHOLDS["high"]:
            return "HIGH"
        elif score >= self.rules.RISK_THRESHOLDS["medium"]:
            return "MEDIUM"
        elif score >= self.rules.RISK_THRESHOLDS["low"]:
            return "LOW"
        return "VERY LOW"
    
    def _create_risk_breakdown(self, category_scores: Dict[str, int],
                              detailed_findings: Dict[str, List[str]]) -> List[RiskBreakdownItem]:
        """Create risk breakdown items for report"""
        breakdown = []
        
        category_names = {
            "restrictive_covenants": "Restrictive Covenants",
            "termination_rights": "Termination Rights",
            "penalties_liability": "Penalties & Liability",
            "compensation_benefits": "Compensation & Benefits",
            "intellectual_property": "Intellectual Property"
        }
        
        for category, score in category_scores.items():
            name = category_names.get(category, category.replace('_', ' ').title())
            findings = detailed_findings.get(category, [])
            
            # Create summary from findings
            if findings:
                summary = ". ".join(findings[:3])  # First 3 findings
            else:
                summary = f"Analysis of {name.lower()} terms and conditions"
            
            breakdown.append(RiskBreakdownItem(
                category=name,
                score=score,
                summary=summary,
                findings=findings
            ))
        
        # Sort by score (highest first)
        breakdown.sort(key=lambda x: x.score, reverse=True)
        
        return breakdown