# DEPENDENCIES
import os
from typing import Any
from io import BytesIO
from typing import Dict
from typing import List
from typing import Optional
from datetime import datetime
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.platypus import Image
from reportlab.platypus import Table
from reportlab.lib.units import inch
from reportlab.platypus import Spacer
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import Paragraph
from reportlab.platypus import PageBreak
from reportlab.graphics import renderPDF
from reportlab.platypus import TableStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.platypus import KeepTogether
from reportlab.graphics.shapes import Circle
from reportlab.graphics.shapes import String
from reportlab.graphics.shapes import Drawing
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate
from reportlab.lib.styles import getSampleStyleSheet



class PDFReportGenerator:
    """
    Generate professional PDF reports matching the sample style
    """
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        

    def _setup_custom_styles(self):
        """
        Setup custom paragraph styles
        """
        # Title style
        self.styles.add(ParagraphStyle(name       = 'ReportTitle',
                                       parent     = self.styles['Heading1'],
                                       fontSize   = 24,
                                       textColor  = colors.HexColor('#1a1a1a'),
                                       spaceAfter = 20,
                                       alignment  = TA_LEFT,
                                       fontName   = 'Helvetica-Bold',
                                      )
                       )
        
        # Section heading
        self.styles.add(ParagraphStyle(name        = 'SectionHeading',
                                       parent      = self.styles['Heading2'],
                                       fontSize    = 16,
                                       textColor   = colors.HexColor('#1a1a1a'),
                                       spaceAfter  = 12,
                                       spaceBefore = 20,
                                       fontName    = 'Helvetica-Bold',
                                      )
                       )
        
        # Body text
        self.styles.add(ParagraphStyle(
            name='CustomBodyText',
            parent=self.styles['Normal'],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor('#333333'),
            alignment=TA_JUSTIFY,
            fontName='Helvetica'
        ))
        
        # Bullet point
        self.styles.add(ParagraphStyle(
            name='BulletPoint',
            parent=self.styles['Normal'],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor('#333333'),
            leftIndent=20,
            bulletIndent=10,
            fontName='Helvetica'
        ))
        
        # Table header
        self.styles.add(ParagraphStyle(
            name='TableHeader',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#1a1a1a'),
            fontName='Helvetica-Bold'
        ))
        
        # Footer
        self.styles.add(ParagraphStyle(
            name='Footer',
            parent=self.styles['Normal'],
            fontSize=8,
            textColor=colors.HexColor('#666666'),
            alignment=TA_CENTER,
            fontName='Helvetica'
        ))
    
    def _draw_risk_score_circle(self, score: int) -> Drawing:
        """Draw the risk score circle graphic"""
        d = Drawing(150, 150)
        
        # Determine color based on score
        if score >= 80:
            color = colors.HexColor('#dc2626')
        elif score >= 60:
            color = colors.HexColor('#f97316')
        elif score >= 40:
            color = colors.HexColor('#ca8a04')
        else:
            color = colors.HexColor('#16a34a')
        
        # Background circle
        bg_circle = Circle(75, 75, 60)
        bg_circle.fillColor = colors.HexColor('#f0f0f0')
        bg_circle.strokeColor = None
        d.add(bg_circle)
        
        # Score circle
        score_circle = Circle(75, 75, 55)
        score_circle.fillColor = color
        score_circle.strokeColor = None
        d.add(score_circle)
        
        # Inner white circle
        inner_circle = Circle(75, 75, 45)
        inner_circle.fillColor = colors.white
        inner_circle.strokeColor = None
        d.add(inner_circle)
        
        # Score text
        score_text = String(75, 70, str(score), textAnchor='middle')
        score_text.fontSize = 36
        score_text.fontName = 'Helvetica-Bold'
        score_text.fillColor = color
        d.add(score_text)
        
        return d
    
    def _get_risk_color(self, score: int) -> colors.Color:
        """Get color based on risk score"""
        if score >= 80:
            return colors.HexColor('#dc2626')
        elif score >= 60:
            return colors.HexColor('#f97316')
        elif score >= 40:
            return colors.HexColor('#ca8a04')
        else:
            return colors.HexColor('#16a34a')
    
    def _create_header_footer(self, canvas, doc):
        """Add header and footer to each page"""
        canvas.saveState()
        
        # Header
        canvas.setFont('Helvetica-Bold', 12)
        canvas.drawString(0.75 * inch, letter[1] - 0.5 * inch, 
                         "AI Contract Risk Analysis Report")
        
        # Footer
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor('#666666'))
        
        # Page number
        page_num = f"Page {doc.page} of {doc.page_count if hasattr(doc, 'page_count') else '?'}"
        canvas.drawString(7 * inch, 0.5 * inch, page_num)
        
        # Legal disclaimer
        disclaimer = "For informational purposes only. Not legal advice."
        canvas.drawCentredString(letter[0] / 2, 0.5 * inch, disclaimer)
        
        canvas.restoreState()
    
    def generate_report(self, analysis_result: Dict[str, Any], 
                       output_path: Optional[str] = None) -> BytesIO:
        """
        Generate PDF report from analysis results
        
        Args:
            analysis_result: Analysis result dictionary from the API
            output_path: Optional file path to save PDF
            
        Returns:
            BytesIO buffer containing the PDF
        """
        # Create buffer
        buffer = BytesIO()
        
        # Create document
        doc = SimpleDocTemplate(
            buffer if not output_path else output_path,
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=1*inch,
            bottomMargin=1*inch
        )
        
        # Build story
        story = []
        
        # Title and Risk Score (Page 1)
        story.extend(self._build_page_1(analysis_result))
        story.append(PageBreak())
        
        # Negotiation Points (Page 2)
        story.extend(self._build_page_2(analysis_result))
        story.append(PageBreak())
        
        # Risk Category Breakdown (Page 3)
        story.extend(self._build_page_3(analysis_result))
        
        # Clause-by-Clause Analysis (Page 4+)
        story.append(PageBreak())
        story.extend(self._build_clause_analysis(analysis_result))
        
        # Build PDF
        doc.build(story, onFirstPage=self._create_header_footer,
                 onLaterPages=self._create_header_footer)
        
        # If using buffer, seek to beginning
        if not output_path:
            buffer.seek(0)
            return buffer
        
        return buffer
    
    def _build_page_1(self, result: Dict) -> List:
        """Build page 1 content: Title, Risk Score, Executive Summary, Key Items"""
        elements = []
        
        # Title
        elements.append(Paragraph("AI Contract Risk Analysis Report", 
                                 self.styles['ReportTitle']))
        elements.append(Spacer(1, 0.1*inch))
        
        # Risk Score Circle
        risk_score = result['risk_analysis']['overall_score']
        elements.append(self._draw_risk_score_circle(risk_score))
        elements.append(Spacer(1, 0.2*inch))
        
        # Executive Summary
        elements.append(Paragraph("Executive Summary", 
                                 self.styles['SectionHeading']))
        elements.append(Paragraph(result['executive_summary'], 
                                 self.styles['BodyText']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Unfavorable Terms
        elements.append(Paragraph("Unfavorable Terms", 
                                 self.styles['SectionHeading']))
        
        for term in result['unfavorable_terms'][:8]:  # Limit to 8 items
            bullet_text = f"<bullet>•</bullet> <b>{term.get('clause_reference', term['term'])}:</b> {term['explanation']}"
            elements.append(Paragraph(bullet_text, self.styles['BulletPoint']))
            elements.append(Spacer(1, 0.05*inch))
        
        elements.append(Spacer(1, 0.2*inch))
        
        # Missing Protections
        elements.append(Paragraph("Missing Protections", 
                                 self.styles['SectionHeading']))
        
        for protection in result['missing_protections'][:6]:  # Limit to 6 items
            bullet_text = f"<bullet>•</bullet> <b>{protection['protection']}:</b> {protection['explanation']}"
            elements.append(Paragraph(bullet_text, self.styles['BulletPoint']))
            elements.append(Spacer(1, 0.05*inch))
        
        return elements
    
    def _build_page_2(self, result: Dict) -> List:
        """Build page 2 content: Negotiation Points"""
        elements = []
        
        elements.append(Paragraph("Negotiation Points", 
                                 self.styles['SectionHeading']))
        elements.append(Spacer(1, 0.1*inch))
        
        negotiation_points = result.get('negotiation_points', [])
        
        if negotiation_points:
            for point in negotiation_points[:7]:  # Limit to 7 points
                bullet_text = f"<bullet>•</bullet> {point['issue']}: {point['rationale']}"
                elements.append(Paragraph(bullet_text, self.styles['BulletPoint']))
                elements.append(Spacer(1, 0.1*inch))
        else:
            # Fallback to unfavorable terms if negotiation points not available
            for term in result['unfavorable_terms'][:7]:
                if term.get('suggested_fix'):
                    bullet_text = f"<bullet>•</bullet> {term['term']}: {term['suggested_fix']}"
                    elements.append(Paragraph(bullet_text, self.styles['BulletPoint']))
                    elements.append(Spacer(1, 0.1*inch))
        
        return elements
    
    def _build_page_3(self, result: Dict) -> List:
        """Build page 3 content: Risk Category Breakdown"""
        elements = []
        
        elements.append(Paragraph("Risk Category Breakdown", 
                                 self.styles['SectionHeading']))
        elements.append(Spacer(1, 0.15*inch))
        
        # Create table data
        table_data = [
            [
                Paragraph('<b>Category</b>', self.styles['TableHeader']),
                Paragraph('<b>Score</b>', self.styles['TableHeader']),
                Paragraph('<b>Summary</b>', self.styles['TableHeader'])
            ]
        ]
        
        risk_breakdown = result['risk_analysis'].get('risk_breakdown', [])
        
        for category in risk_breakdown:
            score_color = self._get_risk_color(category['score'])
            
            category_cell = Paragraph(category['category'], self.styles['BodyText'])
            score_cell = Paragraph(
                f'<font color="{score_color.hexval()}"><b>{category["score"]}</b></font>',
                self.styles['TableHeader']
            )
            summary_cell = Paragraph(category['summary'], self.styles['BodyText'])
            
            table_data.append([category_cell, score_cell, summary_cell])
        
        # Create table
        table = Table(table_data, colWidths=[1.8*inch, 0.7*inch, 4*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f5f5f5')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#1a1a1a')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('TOPPADDING', (0, 1), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e5e5e5')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        
        elements.append(table)
        
        return elements
    
    def _build_clause_analysis(self, result: Dict) -> List:
        """Build clause-by-clause analysis section"""
        elements = []
        
        elements.append(Paragraph("Clause-by-Clause Analysis", 
                                 self.styles['SectionHeading']))
        elements.append(Spacer(1, 0.15*inch))
        
        # Create table data
        table_data = [
            [
                Paragraph('<b>Clause</b>', self.styles['TableHeader']),
                Paragraph('<b>Risk Level</b>', self.styles['TableHeader']),
                Paragraph('<b>Analysis</b>', self.styles['TableHeader']),
                Paragraph('<b>Recommendation</b>', self.styles['TableHeader'])
            ]
        ]
        
        # Get unfavorable terms and interpretations
        unfavorable_terms = result.get('unfavorable_terms', [])
        interpretations = result.get('clause_interpretations', [])
        
        # Combine and process
        processed_clauses = []
        
        for term in unfavorable_terms[:10]:  # Limit to 10 clauses
            clause_ref = term.get('clause_reference', term['term'])
            
            # Find matching interpretation if available
            analysis_text = term['explanation']
            recommendation_text = term.get('suggested_fix', 'Negotiate or seek legal advice.')
            
            # Determine risk level
            severity = term.get('severity', 'high')
            if severity == 'critical':
                risk_level = 'Critical'
                risk_color = colors.HexColor('#dc2626')
            elif severity == 'high':
                risk_level = 'High'
                risk_color = colors.HexColor('#f97316')
            else:
                risk_level = 'Medium'
                risk_color = colors.HexColor('#ca8a04')
            
            clause_cell = Paragraph(clause_ref, self.styles['BodyText'])
            risk_cell = Paragraph(
                f'<font color="{risk_color.hexval()}"><b>{risk_level}</b></font>',
                self.styles['TableHeader']
            )
            analysis_cell = Paragraph(analysis_text, self.styles['BodyText'])
            recommendation_cell = Paragraph(recommendation_text, self.styles['BodyText'])
            
            table_data.append([clause_cell, risk_cell, analysis_cell, recommendation_cell])
        
        # Create table
        table = Table(table_data, colWidths=[1.5*inch, 0.8*inch, 2.2*inch, 2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f5f5f5')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#1a1a1a')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('TOPPADDING', (0, 1), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e5e5e5')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        
        elements.append(table)
        
        return elements


def generate_pdf_report(analysis_result: Dict[str, Any], 
                        output_path: Optional[str] = None) -> BytesIO:
    """
    Convenience function to generate PDF report
    
    Args:
        analysis_result: Complete analysis result from the API
        output_path: Optional file path to save PDF
        
    Returns:
        BytesIO buffer containing the PDF
    """
    generator = PDFReportGenerator()
    return generator.generate_report(analysis_result, output_path)

