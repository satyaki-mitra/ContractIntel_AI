# services/summary_generator.py

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

from utils.logger import ContractAnalyzerLogger
from model_manager.llm_manager import LLMManager, LLMProvider

logger = ContractAnalyzerLogger.get_logger()

@dataclass
class SummaryContext:
    """Context data for summary generation"""
    contract_type: str
    risk_score: int
    risk_level: str
    category_scores: Dict[str, int]
    unfavorable_terms: List[Dict]
    missing_protections: List[Dict]
    clauses: List
    key_findings: List[str]


class SummaryGenerator:
    """
    LLM-powered executive summary generator for contract analysis
    Generates professional, detailed executive summaries like legal professionals
    """
    
    def __init__(self, llm_manager: Optional[LLMManager] = None):
        """
        Initialize the summary generator
        
        Args:
            llm_manager: LLM manager instance (if None, creates one with default settings)
        """
        self.llm_manager = llm_manager or LLMManager()
        self.logger = ContractAnalyzerLogger.get_logger()
        
        # Use proper logging syntax without keyword arguments
        logger.info("Summary generator initialized")
    
    def generate_executive_summary(self, 
                                 classification: Dict,
                                 risk_analysis: Dict,
                                 unfavorable_terms: List[Dict],
                                 missing_protections: List[Dict],
                                 clauses: List) -> str:
        """
        Generate a comprehensive executive summary using LLM
        
        Args:
            classification: Contract classification data
            risk_analysis: Risk analysis results
            unfavorable_terms: List of unfavorable terms
            missing_protections: List of missing protections
            clauses: List of analyzed clauses (ExtractedClause objects)
            
        Returns:
            Generated executive summary string
        """
        try:
            # Prepare context for the LLM
            context = self._prepare_summary_context(
                classification, risk_analysis, unfavorable_terms, 
                missing_protections, clauses
            )
            
            # Generate summary using LLM
            summary = self._generate_with_llm(context)
            
            # Use proper logging syntax
            logger.info(f"Executive summary generated successfully - Risk score: {context.risk_score}, Risk level: {context.risk_level}")
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate executive summary: {e}")
            
            # Create fallback context if preparation failed
            fallback_context = SummaryContext(
                contract_type=classification.get("category", "contract"),
                risk_score=risk_analysis.get("overall_score", 0),
                risk_level=risk_analysis.get("risk_level", "unknown"),
                category_scores=risk_analysis.get("category_scores", {}),
                unfavorable_terms=unfavorable_terms,
                missing_protections=missing_protections,
                clauses=clauses,
                key_findings=[]
            )
            
            # Fallback to simple summary
            return self._generate_fallback_summary(fallback_context)
    
    def _prepare_summary_context(self,
                               classification: Dict,
                               risk_analysis: Dict,
                               unfavorable_terms: List[Dict],
                               missing_protections: List[Dict],
                               clauses: List) -> SummaryContext:
        """Prepare structured context for summary generation"""
        
        contract_type = classification.get("category", "contract")
        risk_score = risk_analysis.get("overall_score", 0)
        risk_level = risk_analysis.get("risk_level", "unknown")
        category_scores = risk_analysis.get("category_scores", {})
        
        # Extract key findings
        key_findings = self._extract_key_findings(
            unfavorable_terms, missing_protections, clauses, risk_score
        )
        
        return SummaryContext(
            contract_type=contract_type,
            risk_score=risk_score,
            risk_level=risk_level,
            category_scores=category_scores,
            unfavorable_terms=unfavorable_terms,
            missing_protections=missing_protections,
            clauses=clauses,
            key_findings=key_findings
        )
    
    def _extract_key_findings(self,
                            unfavorable_terms: List[Dict],
                            missing_protections: List[Dict],
                            clauses: List,
                            risk_score: int) -> List[str]:
        """Extract the most important findings for the summary"""
        
        findings = []
        
        # High-risk clauses - handle both dict and object clauses
        high_risk_clauses = []
        for clause in clauses:
            try:
                # Try to access as object first, then as dict
                if hasattr(clause, 'confidence'):
                    confidence = clause.confidence
                    risk_level = getattr(clause, 'risk_level', None)
                    category = getattr(clause, 'category', 'clause')
                    text = getattr(clause, 'text', '')
                else:
                    # Fallback to dict access
                    confidence = clause.get('confidence', 0)
                    risk_level = clause.get('risk_level')
                    category = clause.get('category', 'clause')
                    text = clause.get('text', '')
                
                if confidence > 0.7 and risk_level in ['high', 'critical']:
                    high_risk_clauses.append({
                        'category': category,
                        'text': text,
                        'confidence': confidence,
                        'risk_level': risk_level
                    })
            except (AttributeError, KeyError, TypeError):
                # Skip clauses that can't be processed
                continue
        
        for clause in high_risk_clauses[:3]:  # Top 3 high-risk clauses
            clause_text = clause['text'][:100] + '...' if len(clause['text']) > 100 else clause['text']
            findings.append(f"High-risk {clause['category']}: {clause_text}")
        
        # Critical unfavorable terms
        critical_terms = []
        for term in unfavorable_terms:
            try:
                if hasattr(term, 'severity'):
                    severity = term.severity
                    term_name = getattr(term, 'term', 'Unknown')
                    explanation = getattr(term, 'explanation', '')
                else:
                    severity = term.get('severity')
                    term_name = term.get('term', 'Unknown')
                    explanation = term.get('explanation', '')
                
                if severity == 'critical':
                    critical_terms.append({
                        'term': term_name,
                        'explanation': explanation
                    })
            except (AttributeError, KeyError, TypeError):
                continue
        
        for term in critical_terms[:2]:
            findings.append(f"Critical term: {term['term']} - {term['explanation']}")
        
        # Important missing protections
        critical_protections = []
        for prot in missing_protections:
            try:
                if hasattr(prot, 'importance'):
                    importance = prot.importance
                    protection_name = getattr(prot, 'protection', 'Unknown')
                    explanation = getattr(prot, 'explanation', '')
                else:
                    importance = prot.get('importance')
                    protection_name = prot.get('protection', 'Unknown')
                    explanation = prot.get('explanation', '')
                
                if importance == 'critical':
                    critical_protections.append({
                        'protection': protection_name,
                        'explanation': explanation
                    })
            except (AttributeError, KeyError, TypeError):
                continue
        
        for prot in critical_protections[:2]:
            findings.append(f"Missing protection: {prot['protection']}")
        
        # Overall risk context
        if risk_score >= 80:
            findings.append("Contract presents critical level of risk requiring immediate attention")
        elif risk_score >= 60:
            findings.append("Significant concerns identified requiring careful review")
        
        return findings
    
    def _generate_with_llm(self, context: SummaryContext) -> str:
        """Generate summary using LLM"""
        
        prompt = self._build_summary_prompt(context)
        system_prompt = self._build_system_prompt()
        
        try:
            response = self.llm_manager.complete(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.3,  # Lower temperature for more consistent, professional output
                max_tokens=800,   # Limit summary length
                json_mode=False
            )
            
            if response.success and response.text.strip():
                return self._clean_summary_response(response.text)
            else:
                raise ValueError(f"LLM generation failed: {response.error_message}")
                
        except Exception as e:
            logger.error(f"LLM summary generation failed: {e}")
            raise
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for professional summary generation"""
        
        return """You are a senior legal analyst specializing in contract risk assessment. Your task is to generate concise, professional executive summaries that:

KEY REQUIREMENTS:
1. Write in formal, professional business language
2. Focus on the most critical risks and implications
3. Be specific about contractual provisions and their impact
4. Maintain objective, factual tone
5. Keep summary length between 100-200 words
6. Structure: Start with overall risk assessment, then key findings, then implications

WRITING STYLE:
- Use precise legal/business terminology
- Avoid markdown formatting
- Be direct and actionable
- Highlight asymmetrical terms and missing protections
- Focus on practical consequences for the signing party

OUTPUT FORMAT:
Return only the executive summary text, no headings, no bullet points, just clean paragraph text."""

    def _build_summary_prompt(self, context: SummaryContext) -> str:
        """Build detailed prompt for summary generation"""
        
        # Build risk context
        risk_context = self._build_risk_context(context)
        
        # Build key provisions section
        key_provisions = self._build_key_provisions_context(context)
        
        # Build missing protections section
        missing_protections_text = self._build_missing_protections_context(context)
        
        prompt = f"""
CONTRACT ANALYSIS DATA:

{risk_context}

{key_provisions}

{missing_protections_text}

GENERATION INSTRUCTIONS:
Based on the analysis above, write a professional executive summary that:
1. Starts with the overall risk assessment for the {context.contract_type}
2. Highlights the 2-3 most critical issues
3. Explains the practical implications for the signing party
4. Mentions any severely imbalanced or punitive clauses
5. Notes significant missing protections

Focus on clarity, specificity, and actionable insights.
"""
        return prompt
    
    def _build_risk_context(self, context: SummaryContext) -> str:
        """Build risk assessment context"""
        
        risk_level_descriptions = {
            "critical": "CRITICAL level of risk requiring immediate attention",
            "high": "HIGH level of risk requiring significant review",
            "medium": "MODERATE level of risk with some concerns",
            "low": "LOW level of risk, generally favorable"
        }
        
        risk_desc = risk_level_descriptions.get(context.risk_level.lower(), "UNKNOWN level of risk")
        
        text = f"RISK ASSESSMENT:\n"
        text += f"- Overall Score: {context.risk_score}/100 ({risk_desc})\n"
        text += f"- Contract Type: {context.contract_type.replace('_', ' ').title()}\n"
        
        # Add category scores
        if context.category_scores:
            text += "- Risk by Category:\n"
            for category, score in context.category_scores.items():
                category_name = category.replace('_', ' ').title()
                text += f"  * {category_name}: {score}/100\n"
        
        return text
    
    def _build_key_provisions_context(self, context: SummaryContext) -> str:
        """Build context about key provisions and unfavorable terms"""
        
        text = "KEY PROVISIONS & UNFAVORABLE TERMS:\n"
        
        # Critical terms first
        critical_terms = []
        for term in context.unfavorable_terms:
            try:
                if hasattr(term, 'severity'):
                    severity = term.severity
                else:
                    severity = term.get('severity')
                
                if severity == 'critical':
                    critical_terms.append(term)
            except (AttributeError, KeyError):
                continue
        
        high_terms = []
        for term in context.unfavorable_terms:
            try:
                if hasattr(term, 'severity'):
                    severity = term.severity
                else:
                    severity = term.get('severity')
                
                if severity == 'high':
                    high_terms.append(term)
            except (AttributeError, KeyError):
                continue
        
        if critical_terms:
            text += f"- Critical Issues Found: {len(critical_terms)}\n"
            for term in critical_terms[:3]:
                try:
                    if hasattr(term, 'term'):
                        term_name = term.term
                        explanation = getattr(term, 'explanation', '')
                    else:
                        term_name = term.get('term', 'Unknown')
                        explanation = term.get('explanation', '')
                    text += f"  * {term_name}: {explanation}\n"
                except (AttributeError, KeyError):
                    continue
        
        if high_terms:
            text += f"- Significant Concerns: {len(high_terms)}\n"
            for term in high_terms[:2]:
                try:
                    if hasattr(term, 'term'):
                        term_name = term.term
                        explanation = getattr(term, 'explanation', '')
                    else:
                        term_name = term.get('term', 'Unknown')
                        explanation = term.get('explanation', '')
                    text += f"  * {term_name}: {explanation}\n"
                except (AttributeError, KeyError):
                    continue
        
        # High-risk clauses
        high_risk_clauses = []
        for clause in context.clauses:
            try:
                if hasattr(clause, 'confidence'):
                    confidence = clause.confidence
                    risk_level = getattr(clause, 'risk_level', None)
                else:
                    confidence = clause.get('confidence', 0)
                    risk_level = clause.get('risk_level')
                
                if confidence > 0.7 and risk_level in ['high', 'critical']:
                    high_risk_clauses.append(clause)
            except (AttributeError, KeyError, TypeError):
                continue
        
        if high_risk_clauses:
            text += f"- High-Risk Clauses Identified: {len(high_risk_clauses)}\n"
            for clause in high_risk_clauses[:2]:
                try:
                    if hasattr(clause, 'category'):
                        category = clause.category
                        clause_text = getattr(clause, 'text', '')
                    else:
                        category = clause.get('category', 'Unknown')
                        clause_text = clause.get('text', '')
                    
                    display_text = clause_text[:80] + '...' if len(clause_text) > 80 else clause_text
                    text += f"  * {category}: {display_text}\n"
                except (AttributeError, KeyError):
                    continue
        
        return text
    
    def _build_missing_protections_context(self, context: SummaryContext) -> str:
        """Build context about missing protections"""
        
        text = "MISSING PROTECTIONS:\n"
        
        critical_protections = []
        for prot in context.missing_protections:
            try:
                if hasattr(prot, 'importance'):
                    importance = prot.importance
                else:
                    importance = prot.get('importance')
                
                if importance == 'critical':
                    critical_protections.append(prot)
            except (AttributeError, KeyError):
                continue
        
        important_protections = []
        for prot in context.missing_protections:
            try:
                if hasattr(prot, 'importance'):
                    importance = prot.importance
                else:
                    importance = prot.get('importance')
                
                if importance == 'high':
                    important_protections.append(prot)
            except (AttributeError, KeyError):
                continue
        
        if critical_protections:
            text += f"- Critical Protections Missing: {len(critical_protections)}\n"
            for prot in critical_protections[:3]:
                try:
                    if hasattr(prot, 'protection'):
                        protection_name = prot.protection
                        explanation = getattr(prot, 'explanation', '')
                    else:
                        protection_name = prot.get('protection', 'Unknown')
                        explanation = prot.get('explanation', '')
                    text += f"  * {protection_name}: {explanation}\n"
                except (AttributeError, KeyError):
                    continue
        
        if important_protections:
            text += f"- Important Protections Missing: {len(important_protections)}\n"
            for prot in important_protections[:2]:
                try:
                    if hasattr(prot, 'protection'):
                        protection_name = prot.protection
                        explanation = getattr(prot, 'explanation', '')
                    else:
                        protection_name = prot.get('protection', 'Unknown')
                        explanation = prot.get('explanation', '')
                    text += f"  * {protection_name}: {explanation}\n"
                except (AttributeError, KeyError):
                    continue
        
        if not critical_protections and not important_protections:
            text += "- No critical protections missing\n"
        
        return text
    
    def _clean_summary_response(self, text: str) -> str:
        """Clean and format the LLM response"""
        
        # Remove any markdown formatting
        text = text.replace('**', '').replace('*', '').replace('#', '')
        
        # Remove common LLM artifacts
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line and not line.lower().startswith(('executive summary', 'summary:', 'here is', 'based on')):
                cleaned_lines.append(line)
        
        # Join into coherent paragraph
        summary = ' '.join(cleaned_lines)
        
        # Ensure proper sentence structure
        if summary and not summary[0].isupper():
            summary = summary[0].upper() + summary[1:]
        
        if summary and not summary.endswith(('.', '!', '?')):
            summary += '.'
        
        return summary
    
    def _generate_fallback_summary(self, context: SummaryContext) -> str:
        """Generate a fallback summary when LLM is not available"""
        
        contract_type_display = context.contract_type.replace('_', ' ').title()
        
        # Count critical items
        critical_terms = 0
        for term in context.unfavorable_terms:
            try:
                if hasattr(term, 'severity'):
                    if term.severity == 'critical':
                        critical_terms += 1
                else:
                    if term.get('severity') == 'critical':
                        critical_terms += 1
            except (AttributeError, KeyError):
                continue
        
        critical_protections = 0
        for prot in context.missing_protections:
            try:
                if hasattr(prot, 'importance'):
                    if prot.importance == 'critical':
                        critical_protections += 1
                else:
                    if prot.get('importance') == 'critical':
                        critical_protections += 1
            except (AttributeError, KeyError):
                continue
        
        if context.risk_score >= 80:
            risk_assessment = f"This {contract_type_display} presents a CRITICAL level of risk"
            action = "requires immediate attention and significant revision"
        elif context.risk_score >= 60:
            risk_assessment = f"This {contract_type_display} presents a HIGH level of risk"
            action = "requires careful review and substantial negotiation"
        elif context.risk_score >= 40:
            risk_assessment = f"This {contract_type_display} presents a MODERATE level of risk"
            action = "requires review and selective negotiation"
        else:
            risk_assessment = f"This {contract_type_display} presents a LOW level of risk"
            action = "appears generally reasonable but should be reviewed"
        
        summary = f"{risk_assessment} with a score of {context.risk_score}/100. "
        summary += f"The agreement {action}. "
        
        if critical_terms > 0:
            summary += f"Found {critical_terms} critical unfavorable terms and "
        else:
            summary += f"Found {len(context.unfavorable_terms)} unfavorable terms and "
        
        if critical_protections > 0:
            summary += f"{critical_protections} critical missing protections. "
        else:
            summary += f"{len(context.missing_protections)} missing protections. "
        
        summary += "Review the detailed analysis below for specific clauses and recommendations."
        
        return summary