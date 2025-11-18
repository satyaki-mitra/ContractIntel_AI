# DEPENDENCIES
import os
import math
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
from reportlab.lib.pagesizes import A4
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import Paragraph
from reportlab.platypus import PageBreak
from reportlab.graphics import renderPDF
from reportlab.platypus import TableStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.graphics.shapes import Path
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import simpleSplit
from reportlab.platypus import KeepTogether
from reportlab.graphics.shapes import Circle
from reportlab.graphics.shapes import String
from reportlab.lib.pagesizes import landscape
from reportlab.graphics.shapes import Drawing
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate
from reportlab.platypus.flowables import PageBreak
from reportlab.platypus.flowables import KeepInFrame
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Table as PlatypusTable


class PDFReportGenerator:
    """
    Professional-grade PDF report generator matching sample style exactly
    """
    def __init__(self):
        self.styles         = getSampleStyleSheet()
        
        self._setup_custom_styles()

        self.page_width     = letter[0]
        self.page_height    = letter[1]
        self.margin_left    = 0.75 * inch
        self.margin_right   = 0.75 * inch
        self.margin_top     = 1.0 * inch
        self.margin_bottom  = 1.0 * inch
        self.content_width  = self.page_width - self.margin_left - self.margin_right
        self.content_height = self.page_height - self.margin_top - self.margin_bottom


    def _setup_custom_styles(self):
        """
        Setup custom paragraph styles with precise control
        """
        # Title style
        self.styles.add(ParagraphStyle(name       = 'ReportTitle',
                                       parent     = self.styles['Heading1'],
                                       fontSize   = 20,
                                       textColor  = colors.HexColor('#1a1a1a'),
                                       spaceAfter = 15,
                                       alignment  = TA_CENTER,  
                                       fontName   = 'Helvetica-Bold',
                                      )
                       )

        # Section heading
        self.styles.add(ParagraphStyle(name        = 'SectionHeading',
                                       parent      = self.styles['Heading2'],
                                       fontSize    = 14,
                                       textColor   = colors.HexColor('#1a1a1a'),
                                       spaceAfter  = 10,
                                       spaceBefore = 15,
                                       fontName    = 'Helvetica-Bold',
                                      )
                       )

        # Sub-section heading
        self.styles.add(ParagraphStyle(name        = 'SubSectionHeading',
                                       parent      = self.styles['Normal'],
                                       fontSize    = 11,
                                       textColor   = colors.HexColor('#333333'),
                                       spaceAfter  = 6,
                                       spaceBefore = 10,
                                       fontName    = 'Helvetica-Bold',
                                      )
                       )

        # Body text
        self.styles.add(ParagraphStyle(name        = 'CustomBodyText',
                                       parent      = self.styles['Normal'],
                                       fontSize    = 9,
                                       leading     = 12,
                                       textColor   = colors.HexColor('#333333'),
                                       alignment   = TA_JUSTIFY,
                                       fontName    = 'Helvetica',
                                       leftIndent  = 0,
                                       rightIndent = 0,
                                      )
                       )

        # Small text style
        self.styles.add(ParagraphStyle(name      = 'SmallText',
                                       parent    = self.styles['Normal'],
                                       fontSize  = 8,
                                       leading   = 10,
                                       textColor = colors.HexColor('#666666'),
                                       fontName  = 'Helvetica',
                                      )
                       )

        # Bullet point
        self.styles.add(ParagraphStyle(name           = 'BulletPoint',
                                       parent         = self.styles['Normal'],
                                       fontSize       = 9,
                                       leading        = 12,
                                       textColor      = colors.HexColor('#333333'),
                                       leftIndent     = 15,
                                       bulletIndent   = 8,
                                       bulletFontName = 'Helvetica',
                                       bulletFontSize = 9,
                                       bulletColor    = colors.black,
                                       spaceAfter     = 3,
                                       fontName       = 'Helvetica',
                                      )
                       )

        # Table header with text wrapping
        self.styles.add(ParagraphStyle(name          = 'TableHeader',
                                       parent        = self.styles['Normal'],
                                       fontSize      = 8,
                                       textColor     = colors.HexColor('#ffffff'),
                                       fontName      = 'Helvetica-Bold',
                                       alignment     = TA_CENTER,
                                       backColor     = colors.HexColor('#374151'),
                                       borderPadding = 4,
                                       spaceAfter    = 0,
                                       spaceBefore   = 0,
                                       wordWrap      = 'LTR',
                                       leading       = 10,
                                      )
                       )

        # Table cell
        self.styles.add(ParagraphStyle(name        = 'TableCell',
                                       parent      = self.styles['Normal'],
                                       fontSize    = 8,
                                       leading     = 10,
                                       textColor   = colors.HexColor('#333333'),
                                       fontName    = 'Helvetica',
                                       alignment   = TA_LEFT,
                                       wordWrap    = 'LTR',
                                       spaceAfter  = 1,
                                       leftIndent  = 2,
                                       rightIndent = 2,
                                      )
                       )

        # Table cell small
        self.styles.add(ParagraphStyle(name       = 'TableCellSmall',
                                       parent     = self.styles['Normal'],
                                       fontSize   = 7,
                                       leading    = 9,
                                       textColor  = colors.HexColor('#333333'),
                                       fontName   = 'Helvetica',
                                       alignment  = TA_LEFT,
                                       wordWrap   = 'LTR',
                                       spaceAfter = 1,
                                      )
                       )

        # Footer
        self.styles.add(ParagraphStyle(name      = 'Footer',
                                       parent    = self.styles['Normal'],
                                       fontSize  = 7,
                                       textColor = colors.HexColor('#666666'),
                                       alignment = TA_CENTER,
                                       fontName  = 'Helvetica',
                                      )
                       )

        # Risk indicator style
        self.styles.add(ParagraphStyle(name          = 'RiskIndicator',
                                       parent        = self.styles['Normal'],
                                       fontSize      = 8,
                                       textColor     = colors.HexColor('#dc2626'),
                                       fontName      = 'Helvetica-Bold',
                                       backColor     = colors.HexColor('#fef2f2'),
                                       borderPadding = 3,
                                       spaceAfter    = 3,
                                      )
                       )

        # Keyword style 
        self.styles.add(ParagraphStyle(name          = 'Keyword',
                                       parent        = self.styles['Normal'],
                                       fontSize      = 6,
                                       textColor     = colors.HexColor('#1e40af'),
                                       fontName      = 'Helvetica-Bold',
                                       backColor     = colors.HexColor('#eff6ff'),
                                       borderPadding = 1,
                                       borderColor   = colors.HexColor('#1e40af'),
                                       borderWidth   = 0.5,
                                       alignment     = TA_CENTER,
                                       spaceAfter    = 1,
                                       spaceBefore   = 1,
                                       leftIndent    = 1,
                                       rightIndent   = 1,
                                      )
                       )

        # Statistics style
        self.styles.add(ParagraphStyle(name       = 'Statistics',
                                       parent     = self.styles['Normal'],
                                       fontSize   = 10,
                                       textColor  = colors.HexColor('#1a1a1a'),
                                       fontName   = 'Helvetica-Bold',
                                       alignment  = TA_LEFT,
                                       spaceAfter = 4,
                                      )
                       )


    def _draw_risk_score_circle(self, score: int) -> Drawing:
        """
        Draw the risk score circle graphic with correct fill percentage
        """
        d                  = Drawing(140, 140)
        
        center_x, center_y = 70, 70
        outer_radius       = 55
        inner_radius       = 40
        thickness          = 15

        if (score >= 80):
            color = colors.HexColor('#dc2626')

        elif (score >= 60):
            color = colors.HexColor('#f97316')

        elif (score >= 40):
            color = colors.HexColor('#ca8a04')
        
        else:
            color = colors.HexColor('#16a34a')

        bg_circle             = Circle(center_x, center_y, outer_radius)
        bg_circle.fillColor   = colors.HexColor('#f0f0f0')
        bg_circle.strokeColor = None
        d.add(bg_circle)

        sweep_angle           = (score / 100.0) * 360
        start_angle           = 90
        end_angle             = start_angle - sweep_angle
        extent                = -sweep_angle

        p                     = Path()
        start_rad             = math.radians(start_angle)
        start_outer_x         = center_x + outer_radius * math.cos(start_rad)
        start_outer_y         = center_y + outer_radius * math.sin(start_rad)
        p.moveTo(start_outer_x, start_outer_y)
        
        num_segments          = max(10, int(sweep_angle / 5))
        angle_step            = sweep_angle / num_segments

        for i in range(1, num_segments + 1):
            current_angle_deg = start_angle - (i * angle_step)
            current_angle_rad = math.radians(current_angle_deg)
            x                 = center_x + outer_radius * math.cos(current_angle_rad)
            y                 = center_y + outer_radius * math.sin(current_angle_rad)
            p.lineTo(x, y)

        for i in range(num_segments, -1, -1):
            current_angle_deg = start_angle - (i * angle_step)
            current_angle_rad = math.radians(current_angle_deg)
            x                 = center_x + inner_radius * math.cos(current_angle_rad)
            y                 = center_y + inner_radius * math.sin(current_angle_rad)
            p.lineTo(x, y)

        p.closePath()
        p.fillColor   = color
        p.strokeColor = None
        d.add(p)

        inner_circle             = Circle(center_x, center_y, inner_radius - 2)
        inner_circle.fillColor   = colors.white
        inner_circle.strokeColor = None
        d.add(inner_circle)

        score_text               = String(center_x, center_y - 12, str(score), textAnchor='middle')
        score_text.fontSize      = 36
        score_text.fontName      = 'Helvetica-Bold'
        score_text.fillColor     = color
        d.add(score_text)

        subtitle_text            = String(center_x, center_y - 30, "/100", textAnchor='middle')
        subtitle_text.fontSize   = 16
        subtitle_text.fontName   = 'Helvetica'
        subtitle_text.fillColor  = colors.HexColor('#666666')
        d.add(subtitle_text)

        return d


    def _get_risk_color(self, score: int) -> colors.Color:
        """
        Get color based on risk score
        """
        if (score >= 80):
            return colors.HexColor('#dc2626')

        elif (score >= 60):
            return colors.HexColor('#f97316')

        elif (score >= 40):
            return colors.HexColor('#ca8a04')

        else:
            return colors.HexColor('#16a34a')


    def _create_header_footer(self, canvas, doc):
        """
        Add header and footer to each page with consistent positioning
        """
        canvas.saveState()

        canvas.setFont('Helvetica-Bold', 7)
        canvas.setFillColor(colors.black)
        canvas.drawString(self.margin_left, self.page_height - 0.7 * inch, "AI Powered Contract Risk Analysis Report")

        canvas.setFont('Helvetica', 7)
        canvas.setFillColor(colors.HexColor('#666666'))

        page_num   = f"Page {doc.page}"
        canvas.drawString(self.page_width - self.margin_right - 0.8*inch, 0.6 * inch, page_num)

        disclaimer = "For informational purposes only. Not legal advice."
        canvas.drawCentredString(self.page_width / 2.0, 0.6 * inch, disclaimer)

        canvas.restoreState()


    def generate_report(self, analysis_result: Dict[str, Any], output_path: Optional[str] = None) -> BytesIO:
        """
        Generate PDF report from analysis results
        """
        buffer = BytesIO()

        doc    = SimpleDocTemplate(buffer if not output_path else output_path,
                                   pagesize     = letter,
                                   rightMargin  = self.margin_right,
                                   leftMargin   = self.margin_left,
                                   topMargin    = self.margin_top,
                                   bottomMargin = self.margin_bottom,
                                  )

        story  = list()

        story.extend(self._build_page_1(analysis_result))
        story.append(PageBreak())

        story.extend(self._build_page_2(analysis_result))
        story.append(PageBreak())

        story.extend(self._build_page_3(analysis_result))
        story.append(PageBreak())

        story.extend(self._build_page_4(analysis_result))
        story.append(PageBreak())

        # Clause Interpretations as Table
        story.extend(self._build_clause_interpretations_table(analysis_result))
        story.append(PageBreak())

        # Detailed Clause Analysis as Table
        story.extend(self._build_detailed_clause_analysis_table(analysis_result))

        doc.build(story, onFirstPage = self._create_header_footer, onLaterPages = self._create_header_footer)

        if not output_path:
            buffer.seek(0)
            return buffer

        return buffer


    def _build_page_1(self, result: Dict) -> List:
        """
        Build page 1 content: Contract info, Risk Score, Statistics, Keywords
        """
        elements = list()

        # Title of the PDF
        elements.append(Paragraph("AI Contract Risk Analysis Report", self.styles['ReportTitle']))
        elements.append(Spacer(1, 0.05*inch))

        # Contract Information Line
        classification     = result.get('classification', {})
        primary_category   = classification.get('category', 'Employment').title()
        subcategory        = classification.get('subcategory', 'Executive').title()
        confidence         = classification.get('confidence', 0) * 100
        detected_keywords  = classification.get('detected_keywords', [])
        keyword_count      = len(detected_keywords)
        
        contract_info_text = f"<b>Contract Category:</b> {primary_category} | <b>Contract Subcategory:</b> {subcategory} | <b>Confidence:</b> {confidence:.1f}% | <b>Identified Keywords:</b> {keyword_count}"
        elements.append(Paragraph(contract_info_text, self.styles['CustomBodyText']))
        elements.append(Spacer(1, 0.15*inch))

        # Single line with risk circle and score info
        risk_analysis  = result.get('risk_analysis', {})
        overall_score  = risk_analysis.get('overall_score', 0)
        risk_level     = risk_analysis.get('risk_level', 'UNKNOWN')
        
        # Create risk circle and score info side by side
        score_frame    = KeepInFrame(1.6*inch, 1.6*inch, [self._draw_risk_score_circle(overall_score)])
        
        # Just the score info next to the circle
        risk_info_text = f"<b>Overall Risk Score: {overall_score}/100 ({risk_level})</b>"
        risk_info_para = Paragraph(risk_info_text, self.styles['CustomBodyText'])

        # Create side-by-side layout for risk circle and score
        risk_layout = PlatypusTable([[score_frame, risk_info_para]], colWidths = [1.7*inch, 4.0*inch])
        risk_layout.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                                         ('LEFTPADDING', (0, 0), (-1, -1), 0),
                                         ('RIGHTPADDING', (0, 0), (-1, -1), 0),
                                         ('TOPPADDING', (0, 0), (-1, -1), 0),
                                         ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
                                       ])
                            )
        
        elements.append(risk_layout)
        elements.append(Spacer(1, 0.2*inch))

        # Contract Summary Section - Separate section below risk circle
        elements.append(Paragraph("Contract Summary", self.styles['SectionHeading']))
        executive_summary = result.get('executive_summary', 'No executive summary available.')
        summary_para      = Paragraph(executive_summary, self.styles['CustomBodyText'])
        elements.append(summary_para)
        elements.append(Spacer(1, 0.2*inch))

        # Statistics Section
        elements.append(Paragraph("Contract Statistics", self.styles['SectionHeading']))
        
        # Calculate statistics
        clauses                = result.get('clauses', [])
        total_clauses          = len(clauses)
        risky_clauses          = len([c for c in clauses if c.get('risk_score', 0) > 50])
        unfavorable_terms      = len(result.get('unfavorable_terms', []))
        missing_protections    = len(result.get('missing_protections', []))
        clause_interpretations = len(result.get('clause_interpretations', []))
        
        # Create statistics table
        stats_data             = [[Paragraph('<b>Metric</b>', self.styles['TableHeader']), Paragraph('<b>Count</b>', self.styles['TableHeader'])],
                                  [Paragraph("Total Clauses Analyzed", self.styles['TableCell']), Paragraph(str(total_clauses), self.styles['TableCell'])],
                                  [Paragraph("High-Risk Clauses", self.styles['TableCell']), Paragraph(str(risky_clauses), self.styles['TableCell'])],
                                  [Paragraph("Unfavorable Terms", self.styles['TableCell']), Paragraph(str(unfavorable_terms), self.styles['TableCell'])],
                                  [Paragraph("Missing Protections", self.styles['TableCell']), Paragraph(str(missing_protections), self.styles['TableCell'])],
                                  [Paragraph("Clause Interpretations", self.styles['TableCell']), Paragraph(str(clause_interpretations), self.styles['TableCell'])],
                                 ]
        
        stats_table            = Table(stats_data, colWidths=[2.5*inch, 1*inch])
        
        stats_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#374151')),
                                         ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                                         ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                                         ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                         ('FONTSIZE', (0, 0), (-1, -1), 9),
                                         ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                                         ('TOPPADDING', (0, 1), (-1, -1), 6),
                                         ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
                                         ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#d1d5db')),
                                         ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                                         ('LEFTPADDING', (0, 0), (-1, -1), 8),
                                         ('RIGHTPADDING', (0, 0), (-1, -1), 8),
                                       ])
                            )
        
        elements.append(stats_table)
        elements.append(Spacer(1, 0.2*inch))

        # Keywords and Reasoning Section
        elements.append(Paragraph("Keywords Analysis", self.styles['SectionHeading']))
        
        # Reasoning
        reasoning = classification.get('reasoning', [])
        
        if reasoning:
            elements.append(Paragraph("<b>Classification Reasoning:</b>", self.styles['CustomBodyText']))
            for reason in reasoning:
                elements.append(Paragraph(f"• {reason}", self.styles['BulletPoint']))
            
            elements.append(Spacer(1, 0.1*inch))

        # Detected Keywords 
        if detected_keywords:
            elements.append(Paragraph("<b>Detected Top Keywords:</b>", self.styles['CustomBodyText']))
            elements.append(Spacer(1, 0.05*inch))
            
            # Create keyword tags in a table format
            keywords_per_row  = 7
            first_14_keywords = detected_keywords[:14]
            keyword_chunks    = [first_14_keywords[i:i + keywords_per_row] for i in range(0, len(first_14_keywords), keywords_per_row)]
            
            for chunk in keyword_chunks:
                # Create a row of keyword tags 
                keyword_cells = []
                for keyword in chunk:
                    keyword_tag = Paragraph(str(keyword), self.styles['Keyword'])
                    keyword_cells.append(keyword_tag)
                
                # Create table with only the actual keyword cells
                keyword_row = Table([keyword_cells], colWidths = [1.0*inch] * len(keyword_cells))
                
                keyword_row.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                                                 ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                                                 ('LEFTPADDING', (0, 0), (-1, -1), 2),
                                                 ('RIGHTPADDING', (0, 0), (-1, -1), 2),
                                                 ('TOPPADDING', (0, 0), (-1, -1), 2),
                                                 ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
                                               ])
                                    )

                elements.append(keyword_row)
                elements.append(Spacer(1, 0.05*inch))
        
        return elements


    def _build_page_2(self, result: Dict) -> List:
        """
        Build page 2: Unfavorable Terms and Missing Protections as TABLES
        """
        elements    = list()

        # Unfavorable Terms as TABLE
        elements.append(Paragraph("Unfavorable Terms", self.styles['SectionHeading']))
        
        unfav_terms = result.get('unfavorable_terms', [])

        if unfav_terms:
            # Prepare table data
            table_data   = [[Paragraph('<b>Clause<br/>Reference</b>', self.styles['TableHeader']),
                             Paragraph('<b>Severity</b>', self.styles['TableHeader']),
                             Paragraph('<b>Risk<br/>Score</b>', self.styles['TableHeader']),
                             Paragraph('<b>Explanation</b>', self.styles['TableHeader']),
                           ]]
            
            sorted_terms = sorted(unfav_terms, key = lambda x: (x.get('severity', 'low') != 'high', -x.get('risk_score', 0)))
            
            for term in sorted_terms:
                severity         = term.get('severity', 'unknown').upper()
                risk_score       = term.get('risk_score', 0)
                clause_ref       = term.get('clause_reference', 'N/A') or 'N/A' 
                explanation      = term.get('explanation', 'No explanation provided.') or 'No explanation provided.' 
                
                severity_color   = self._get_severity_color(severity)
                
                clause_ref_para  = Paragraph(str(clause_ref), self.styles['TableCell'])  
                severity_para    = Paragraph(f'<font color="{severity_color.hexval()}">{severity}</font>', self.styles['TableCell'])
                risk_score_para  = Paragraph(str(risk_score), self.styles['TableCell'])
                explanation_para = Paragraph(str(explanation), self.styles['TableCell']) 
                
                table_data.append([clause_ref_para, severity_para, risk_score_para, explanation_para])

            col_widths = [1.2*inch, 1.2*inch, 1.0*inch, 3.0*inch]
            table      = Table(table_data, colWidths = col_widths, repeatRows = 1)
            
            table.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.HexColor('#374151')),
                                       ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                                       ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                                       ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                                       ('FONTSIZE', (0,0), (-1,0), 8),
                                       ('BOTTOMPADDING', (0,0), (-1,0), 8),
                                       ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#d1d5db')),
                                       ('VALIGN', (0,0), (-1,-1), 'TOP'),
                                       ('LEFTPADDING', (0,0), (-1,-1), 6),
                                       ('RIGHTPADDING', (0,0), (-1,-1), 6),
                                       ('TOPPADDING', (0,0), (-1,-1), 6),
                                       ('BOTTOMPADDING', (0,0), (-1,-1), 6),
                                       ('WORDWRAP', (0,0), (-1,-1), True),
                                     ])
                          )

            elements.append(table)

        else:
            elements.append(Paragraph("No unfavorable terms identified.", self.styles['CustomBodyText']))

        elements.append(Spacer(1, 0.2*inch))
        
        # Missing Protections as TABLE
        elements.append(Paragraph("Missing Protections", self.styles['SectionHeading']))
        missing_protections = result.get('missing_protections', [])
        
        if missing_protections:
            # Prepare table data
            table_data         = [[Paragraph('<b>Protection<br/>Name</b>', self.styles['TableHeader']),
                                   Paragraph('<b>Importance</b>', self.styles['TableHeader']),
                                   Paragraph('<b>Risk<br/>Score</b>', self.styles['TableHeader']),
                                   Paragraph('<b>Explanation</b>', self.styles['TableHeader']),
                                 ]]
            
            sorted_protections = sorted(missing_protections, key = lambda x: (x.get('importance', 'medium') != 'critical', -x.get('risk_score', 0)))
            
            for prot in sorted_protections:
                importance       = prot.get('importance', 'medium').upper()
                risk_score       = prot.get('risk_score', 0)
                protection_name  = prot.get('protection', 'N/A') or 'N/A' 
                explanation      = prot.get('explanation', 'No explanation provided.') or 'No explanation provided.' 
                
                importance_color = self._get_importance_color(importance)
                
                protection_para  = Paragraph(str(protection_name), self.styles['TableCell']) 
                importance_para  = Paragraph(f'<font color="{importance_color.hexval()}">{importance}</font>', self.styles['TableCell'])
                risk_score_para  = Paragraph(str(risk_score), self.styles['TableCell'])
                explanation_para = Paragraph(str(explanation), self.styles['TableCell'])  
                
                table_data.append([protection_para, importance_para, risk_score_para, explanation_para])

            col_widths = [1.5*inch, 1.5*inch, 1.2*inch, 3.0*inch]
            table      = Table(table_data, colWidths = col_widths, repeatRows = 1)
            
            table.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.HexColor('#374151')),
                                       ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                                       ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                                       ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                                       ('FONTSIZE', (0,0), (-1,0), 8),
                                       ('BOTTOMPADDING', (0,0), (-1,0), 8),
                                       ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#d1d5db')),
                                       ('VALIGN', (0,0), (-1,-1), 'TOP'),
                                       ('LEFTPADDING', (0,0), (-1,-1), 6),
                                       ('RIGHTPADDING', (0,0), (-1,-1), 6),
                                       ('TOPPADDING', (0,0), (-1,-1), 6),
                                       ('BOTTOMPADDING', (0,0), (-1,-1), 6),
                                       ('WORDWRAP', (0,0), (-1,-1), True),
                                     ])
                          )

            elements.append(table)

        else:
            elements.append(Paragraph("No missing protections identified.", self.styles['CustomBodyText']))

        return elements


    def _build_page_3(self, result: Dict) -> List:
        """
        Build page 3: Negotiation Points as a structured table
        """
        elements           = list()
        
        elements.append(Paragraph("Negotiation Strategy", self.styles['SectionHeading']))
        
        negotiation_points = result.get('negotiation_points', [])

        if negotiation_points:
            table_data = [[Paragraph('<b>Priority</b>', self.styles['TableHeader']),
                           Paragraph('<b>Issue</b>', self.styles['TableHeader']),
                           Paragraph('<b>Current<br/>Language</b>', self.styles['TableHeader']),
                           Paragraph('<b>Proposed<br/>Language</b>', self.styles['TableHeader']),
                         ]]
            
            sorted_points = sorted(negotiation_points, key=lambda x: x.get('priority', 999))
            
            for point in sorted_points:
                priority = str(point.get('priority', 'N/A'))
                issue    = Paragraph(point.get('issue', 'N/A'), self.styles['TableCell'])
                current  = Paragraph(point.get('current_language', 'Not specified'), self.styles['TableCell'])
                proposed = Paragraph(point.get('proposed_language', 'Request balanced language'), self.styles['TableCell'])
                
                table_data.append([Paragraph(priority, self.styles['TableCell']), issue, current, proposed])

            col_widths = [0.8*inch, 1.2*inch, 2.5*inch, 2.5*inch]
            table      = Table(table_data, colWidths = col_widths, repeatRows = 1)
            
            table.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.HexColor('#374151')),
                                       ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                                       ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                                       ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                                       ('FONTSIZE', (0,0), (-1,0), 8),
                                       ('BOTTOMPADDING', (0,0), (-1,0), 8),
                                       ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#d1d5db')),
                                       ('VALIGN', (0,0), (-1,-1), 'TOP'),
                                       ('LEFTPADDING', (0,0), (-1,-1), 6),
                                       ('RIGHTPADDING', (0,0), (-1,-1), 6),
                                       ('TOPPADDING', (0,0), (-1,-1), 6),
                                       ('BOTTOMPADDING', (0,0), (-1,-1), 6),
                                     ])
                          )

            elements.append(table)

        else:
            elements.append(Paragraph("No negotiation points available.", self.styles['CustomBodyText']))

        return elements

    
    def _build_page_4(self, result: Dict) -> List:
        """
        Build page 4: Risk Category Breakdown Table
        """
        elements       = list()
        
        elements.append(Paragraph("Risk Category Breakdown", self.styles['SectionHeading']))
        
        risk_breakdown = result.get('risk_analysis', {}).get('risk_breakdown', [])
        
        if risk_breakdown:
            table_data = [[Paragraph('<b>Category</b>', self.styles['TableHeader']),
                           Paragraph('<b>Score</b>', self.styles['TableHeader']),
                           Paragraph('<b>Summary</b>', self.styles['TableHeader']),
                         ]]

            for item in risk_breakdown:
                category     = item.get('category', 'N/A').replace('_', ' ').title()
                score        = item.get('score', 0)
                summary      = item.get('summary', 'No summary available.')
                score_color  = self._get_risk_color(score)
                score_para   = Paragraph(f'<font color="{score_color.hexval()}">{score}/100</font>', self.styles['TableCell'])
                summary_para = Paragraph(summary, self.styles['TableCell'])
                
                table_data.append([Paragraph(category, self.styles['TableCell']), score_para, summary_para])

            col_widths = [1.5*inch, 1.5*inch, 3.8*inch]
            table      = Table(table_data, colWidths = col_widths, repeatRows = 1)

            table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#374151')),
                                       ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                                       ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                                       ('ALIGN', (1, 0), (1, -1), 'CENTER'),
                                       ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                       ('FONTSIZE', (0, 0), (-1, 0), 8),
                                       ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                                       ('TOPPADDING', (0, 1), (-1, -1), 6),
                                       ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
                                       ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#d1d5db')),
                                       ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                                       ('LEFTPADDING', (0, 0), (-1, -1), 6),
                                       ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                                     ])
                          )

            elements.append(table)

        else:
            elements.append(Paragraph("No risk breakdown data available.", self.styles['CustomBodyText']))
        
        return elements


    def _build_clause_interpretations_table(self, result: Dict) -> List:
        """
        Build clause interpretations as a table with full text
        """
        elements               = list()

        elements.append(Paragraph("Clause Interpretations", self.styles['SectionHeading']))
        
        clause_interpretations = result.get('clause_interpretations', [])

        if clause_interpretations:
            table_data = [[Paragraph('<b>Clause Reference</b>', self.styles['TableHeader']),
                           Paragraph('<b>Favorability</b>', self.styles['TableHeader']),
                           Paragraph('<b>Summary</b>', self.styles['TableHeader']),
                           Paragraph('<b>Key Points</b>', self.styles['TableHeader']),
                         ]]

            for clause in clause_interpretations:
                ref             = clause.get('clause_reference', 'N/A') or 'N/A'  
                plain_english   = clause.get('plain_english_summary', 'No summary available.') or 'No summary available.' 
                favorability    = clause.get('favorability', 'neutral') or 'neutral'  
                
                fav_color       = self._get_favorability_color(favorability)
                
                key_points      = clause.get('key_points', []) or []  
                key_points_text = ""
                
                if key_points:
                    key_points_text = "• " + "<br/>• ".join([str(kp) for kp in key_points]) 
                
                clause_ref_para   = Paragraph(str(ref), self.styles['TableCell']) 
                favorability_para = Paragraph(f"<font color='{fav_color.hexval()}'>{favorability.upper()}</font>", self.styles['TableCell'])
                summary_para      = Paragraph(str(plain_english), self.styles['TableCell']) 
                key_points_para   = Paragraph(str(key_points_text), self.styles['TableCell']) 
                
                table_data.append([clause_ref_para, favorability_para, summary_para, key_points_para])

            col_widths = [1.0*inch, 1.0*inch, 2.5*inch, 2.5*inch]
            table      = Table(table_data, colWidths = col_widths, repeatRows = 1)
            
            table.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.HexColor('#374151')),
                                       ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                                       ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                                       ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                                       ('FONTSIZE', (0,0), (-1,0), 8),
                                       ('BOTTOMPADDING', (0,0), (-1,0), 8),
                                       ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#d1d5db')),
                                       ('VALIGN', (0,0), (-1,-1), 'TOP'),
                                       ('LEFTPADDING', (0,0), (-1,-1), 6),
                                       ('RIGHTPADDING', (0,0), (-1,-1), 6),
                                       ('TOPPADDING', (0,0), (-1,-1), 6),
                                       ('BOTTOMPADDING', (0,0), (-1,-1), 6),
                                       ('WORDWRAP', (0,0), (-1,-1), True),
                                     ])
                          )

            elements.append(table)

        else:
            elements.append(Paragraph("No clause interpretations available.", self.styles['CustomBodyText']))

        return elements


    def _build_detailed_clause_analysis_table(self, result: Dict) -> List:
        """
        Build detailed clause analysis as a table with full text
        """
        elements = list()

        elements.append(Paragraph("Detailed Clause Analysis", self.styles['SectionHeading']))
        
        clauses  = result.get('clauses', [])

        if clauses:
            table_data = [[Paragraph('<b>Reference</b>', self.styles['TableHeader']),
                           Paragraph('<b>Category</b>', self.styles['TableHeader']),
                           Paragraph('<b>Confidence</b>', self.styles['TableHeader']),
                           Paragraph('<b>Risk<br/>Score</b>', self.styles['TableHeader']),
                           Paragraph('<b>Original<br/>Text</b>', self.styles['TableHeader']),
                           Paragraph('<b>Risk<br/>Indicators</b>', self.styles['TableHeader']),
                         ]]

            for clause in clauses:
                ref                  = clause.get('reference', 'N/A') or 'N/A' 
                category             = clause.get('category', 'N/A') or 'N/A'  
                confidence           = clause.get('confidence', 0)
                risk_score           = clause.get('risk_score', 0)
                clause_text          = clause.get('text', 'No text available.') or 'No text available.'  
                risk_indicators      = clause.get('risk_indicators', []) or []  
                
                risk_indicators_text = ", ".join([str(ri) for ri in risk_indicators]) if risk_indicators else "None" 
                
                ref_para             = Paragraph(str(ref), self.styles['TableCell']) 
                category_para        = Paragraph(str(category).replace('_', ' ').title(), self.styles['TableCell'])  
                confidence_para      = Paragraph(f"{confidence:.1f}", self.styles['TableCell'])
                risk_score_para      = Paragraph(f"{risk_score:.1f}", self.styles['TableCell'])
                text_para            = Paragraph(str(clause_text), self.styles['TableCell'])  
                indicators_para      = Paragraph(str(risk_indicators_text), self.styles['TableCell']) 
                
                table_data.append([ref_para, category_para, confidence_para, risk_score_para, text_para, indicators_para])

            col_widths = [0.8*inch, 0.8*inch, 0.8*inch, 1.0*inch, 3.0*inch, 1.0*inch]
            table      = Table(table_data, colWidths = col_widths, repeatRows = 1)
            
            table.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.HexColor('#374151')),
                                       ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                                       ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                                       ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                                       ('FONTSIZE', (0,0), (-1,0), 8),
                                       ('BOTTOMPADDING', (0,0), (-1,0), 8),
                                       ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#d1d5db')),
                                       ('VALIGN', (0,0), (-1,-1), 'TOP'),
                                       ('LEFTPADDING', (0,0), (-1,-1), 6),
                                       ('RIGHTPADDING', (0,0), (-1,-1), 6),
                                       ('TOPPADDING', (0,0), (-1,-1), 6),
                                       ('BOTTOMPADDING', (0,0), (-1,-1), 6),
                                       ('WORDWRAP', (0,0), (-1,-1), True),
                                     ])
                          )

            elements.append(table)

        else:
            elements.append(Paragraph("No detailed clause analysis available.", self.styles['CustomBodyText']))

        return elements


    def _get_severity_color(self, severity: str) -> colors.Color:
        """
        Get color based on severity level
        """
        severity = severity.lower()
        
        if (severity == 'high'):
            return colors.HexColor('#dc2626')

        elif (severity == 'medium'):
            return colors.HexColor('#f97316')

        else:
            return colors.HexColor('#16a34a')


    def _get_importance_color(self, importance: str) -> colors.Color:
        """
        Get color based on importance level
        """
        importance = importance.lower()
        
        if (importance == 'critical'):
            return colors.HexColor('#dc2626')

        elif (importance == 'high'):
            return colors.HexColor('#f97316')

        elif (importance == 'medium'):
            return colors.HexColor('#ca8a04')

        else:
            return colors.HexColor('#16a34a')


    def _get_favorability_color(self, favorability: str) -> colors.Color:
        """
        Get color based on favorability
        """
        favorability = favorability.lower()
        
        if (favorability == 'favorable'):
            return colors.HexColor('#16a34a')

        elif (favorability == 'unfavorable'):
            return colors.HexColor('#dc2626')

        else:
            return colors.HexColor('#ca8a04')



def generate_pdf_report(analysis_result: Dict[str, Any], output_path: Optional[str] = None) -> BytesIO:
    """
    Convenience function to generate PDF report
    """
    generator        = PDFReportGenerator()

    generator_buffer = generator.generate_report(analysis_result = analysis_result, 
                                                 output_path     = output_path,
                                                )

    return generator_buffer
