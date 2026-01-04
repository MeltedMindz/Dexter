#!/usr/bin/env python3
"""
Create Dexter Protocol Pitch Deck PDF
"""

from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib.colors import HexColor, white, black
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas
from reportlab.lib import colors
import os

# Brand colors
DEXTER_PURPLE = HexColor('#7C3AED')
DEXTER_DARK = HexColor('#1F2937')
DEXTER_LIGHT_PURPLE = HexColor('#A78BFA')
DEXTER_GRAY = HexColor('#6B7280')

class DexterPitchDeck:
    def __init__(self, filename='dexter_pitch_deck.pdf'):
        self.filename = filename
        self.width, self.height = landscape(letter)
        self.doc = SimpleDocTemplate(
            filename,
            pagesize=landscape(letter),
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=1*inch,
            bottomMargin=0.75*inch
        )
        self.story = []
        self.styles = getSampleStyleSheet()
        self._create_custom_styles()
        
    def _create_custom_styles(self):
        # Title style
        self.styles.add(ParagraphStyle(
            name='DexterTitle',
            parent=self.styles['Title'],
            fontSize=42,
            textColor=DEXTER_PURPLE,
            alignment=TA_CENTER,
            spaceAfter=30,
            fontName='Helvetica-Bold'
        ))
        
        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='DexterSubtitle',
            parent=self.styles['Normal'],
            fontSize=24,
            textColor=DEXTER_DARK,
            alignment=TA_CENTER,
            spaceAfter=20,
            fontName='Helvetica'
        ))
        
        # Section header
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading1'],
            fontSize=32,
            textColor=DEXTER_PURPLE,
            alignment=TA_LEFT,
            spaceAfter=20,
            fontName='Helvetica-Bold'
        ))
        
        # Body text
        self.styles.add(ParagraphStyle(
            name='DexterBody',
            parent=self.styles['Normal'],
            fontSize=16,
            textColor=DEXTER_DARK,
            alignment=TA_LEFT,
            spaceAfter=12,
            leading=24,
            fontName='Helvetica'
        ))
        
        # Bullet points
        self.styles.add(ParagraphStyle(
            name='BulletPoint',
            parent=self.styles['Normal'],
            fontSize=16,
            textColor=DEXTER_DARK,
            alignment=TA_LEFT,
            leftIndent=20,
            spaceAfter=10,
            leading=22,
            fontName='Helvetica'
        ))
        
        # Large number style
        self.styles.add(ParagraphStyle(
            name='LargeNumber',
            parent=self.styles['Normal'],
            fontSize=48,
            textColor=DEXTER_PURPLE,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
    def add_title_slide(self):
        self.story.append(Spacer(1, 2*inch))
        self.story.append(Paragraph("DEXTER PROTOCOL", self.styles['DexterTitle']))
        self.story.append(Paragraph("AI-Powered DeFi Liquidity Management", self.styles['DexterSubtitle']))
        self.story.append(Spacer(1, 0.5*inch))
        self.story.append(Paragraph("$200,000 Funding Round | $1M Valuation Cap", self.styles['DexterBody']))
        self.story.append(Paragraph("Q3 2025 Mainnet Launch", self.styles['DexterBody']))
        self.story.append(PageBreak())
        
    def add_problem_slide(self):
        self.story.append(Paragraph("The Problem", self.styles['SectionHeader']))
        self.story.append(Spacer(1, 0.5*inch))
        self.story.append(Paragraph("<b>89%</b> of Uniswap V3 LPs Underperform", self.styles['DexterSubtitle']))
        self.story.append(Spacer(1, 0.5*inch))
        self.story.append(Paragraph("• Capital Inefficiency", self.styles['BulletPoint']))
        self.story.append(Paragraph("• Active Management Burden", self.styles['BulletPoint']))
        self.story.append(Paragraph("• Institutional Barriers", self.styles['BulletPoint']))
        self.story.append(Paragraph("• Fragmented Solutions", self.styles['BulletPoint']))
        self.story.append(PageBreak())
        
    def add_solution_slide(self):
        self.story.append(Paragraph("Our Solution", self.styles['SectionHeader']))
        self.story.append(Spacer(1, 0.3*inch))
        self.story.append(Paragraph("AI-Native Vault Infrastructure", self.styles['DexterSubtitle']))
        self.story.append(Spacer(1, 0.3*inch))
        
        # Create two-column layout for features
        features_data = [
            ["• LSTM Price Prediction", "• Multi-Range Management"],
            ["• Hybrid Strategy Modes", "• Tiered Fee Structure"],
            ["• ERC4626 Compliant", "• Enterprise Security"]
        ]
        
        features_table = Table(features_data, colWidths=[4*inch, 4*inch])
        features_table.setStyle(TableStyle([
            ('FONT', (0, 0), (-1, -1), 'Helvetica', 16),
            ('TEXTCOLOR', (0, 0), (-1, -1), DEXTER_DARK),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 12),
            ('RIGHTPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        
        self.story.append(features_table)
        self.story.append(PageBreak())
        
    def add_market_opportunity_slide(self):
        self.story.append(Paragraph("Market Opportunity", self.styles['SectionHeader']))
        self.story.append(Spacer(1, 0.3*inch))
        
        market_data = [
            ["TAM", "$120B+", "Total DeFi TVL"],
            ["SAM", "$25B", "Concentrated Liquidity"],
            ["SOM", "$1.5B", "Actively Managed"],
            ["Base TVL", "$8B+", "Growing Ecosystem"]
        ]
        
        market_table = Table(market_data, colWidths=[2*inch, 2*inch, 4*inch])
        market_table.setStyle(TableStyle([
            ('FONT', (0, 0), (0, -1), 'Helvetica-Bold', 18),
            ('FONT', (1, 0), (1, -1), 'Helvetica-Bold', 24),
            ('FONT', (2, 0), (2, -1), 'Helvetica', 16),
            ('TEXTCOLOR', (0, 0), (0, -1), DEXTER_PURPLE),
            ('TEXTCOLOR', (1, 0), (1, -1), DEXTER_PURPLE),
            ('TEXTCOLOR', (2, 0), (2, -1), DEXTER_GRAY),
            ('ALIGN', (0, 0), (1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 12),
            ('RIGHTPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 15),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
        ]))
        
        self.story.append(market_table)
        self.story.append(PageBreak())
        
    def add_competitive_advantages_slide(self):
        self.story.append(Paragraph("Competitive Advantages", self.styles['SectionHeader']))
        self.story.append(Spacer(1, 0.5*inch))
        
        self.story.append(Paragraph("✓ First AI-Native ERC4626 Protocol", self.styles['BulletPoint']))
        self.story.append(Paragraph("✓ LSTM Neural Networks for Price Prediction", self.styles['BulletPoint']))
        self.story.append(Paragraph("✓ Performance-Aligned Fee Structure", self.styles['BulletPoint']))
        self.story.append(Paragraph("✓ Base-Native Architecture", self.styles['BulletPoint']))
        self.story.append(Paragraph("✓ Institutional-Grade Security", self.styles['BulletPoint']))
        self.story.append(PageBreak())
        
    def add_roadmap_slide(self):
        self.story.append(Paragraph("Roadmap", self.styles['SectionHeader']))
        self.story.append(Spacer(1, 0.3*inch))
        
        roadmap_data = [
            ["Q3 2025", "Security & Launch", "Smart contract audits • Mainnet deployment • Initial partnerships"],
            ["Q4 2025", "Growth & Expansion", "Cross-chain deployment • Advanced ML features • Institutional onboarding"],
            ["2026", "Platform Maturity", "DAO governance • Advanced strategies • Market leadership"]
        ]
        
        roadmap_table = Table(roadmap_data, colWidths=[1.2*inch, 2.3*inch, 4.5*inch])
        roadmap_table.setStyle(TableStyle([
            ('FONT', (0, 0), (0, -1), 'Helvetica-Bold', 14),
            ('FONT', (1, 0), (1, -1), 'Helvetica-Bold', 14),
            ('FONT', (2, 0), (2, -1), 'Helvetica', 12),
            ('TEXTCOLOR', (0, 0), (0, -1), DEXTER_PURPLE),
            ('TEXTCOLOR', (1, 0), (1, -1), DEXTER_DARK),
            ('TEXTCOLOR', (2, 0), (2, -1), DEXTER_GRAY),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 12),
            ('RIGHTPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ]))
        
        self.story.append(roadmap_table)
        self.story.append(PageBreak())
        
    def add_funding_slide(self):
        self.story.append(Paragraph("Investment Opportunity", self.styles['SectionHeader']))
        self.story.append(Spacer(1, 0.5*inch))
        
        funding_data = [
            ["Raise Amount", "$200,000"],
            ["Valuation Cap", "$1M"],
            ["Structure", "SAFE"],
            ["Early Discount", "15%"],
            ["Use of Funds", "MVP Development & Launch"]
        ]
        
        funding_table = Table(funding_data, colWidths=[3*inch, 3*inch])
        funding_table.setStyle(TableStyle([
            ('FONT', (0, 0), (0, -1), 'Helvetica', 18),
            ('FONT', (1, 0), (1, -1), 'Helvetica-Bold', 20),
            ('TEXTCOLOR', (0, 0), (0, -1), DEXTER_GRAY),
            ('TEXTCOLOR', (1, 0), (1, -1), DEXTER_PURPLE),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 12),
            ('RIGHTPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ]))
        
        self.story.append(funding_table)
        self.story.append(PageBreak())
        
    def add_team_slide(self):
        self.story.append(Paragraph("Team", self.styles['SectionHeader']))
        self.story.append(Spacer(1, 0.3*inch))
        
        # Center the team member info
        team_style = ParagraphStyle(
            name='TeamCenter',
            parent=self.styles['DexterSubtitle'],
            alignment=TA_CENTER
        )
        
        role_style = ParagraphStyle(
            name='RoleCenter',
            parent=self.styles['DexterBody'],
            alignment=TA_CENTER,
            fontSize=18,
            textColor=DEXTER_GRAY
        )
        
        self.story.append(Paragraph("<b>Melted</b>", team_style))
        self.story.append(Paragraph("Founder & Technical Lead", role_style))
        self.story.append(Spacer(1, 0.5*inch))
        
        # Create expertise table for better formatting
        expertise_data = [
            ["Built Complete DeFi Protocol"],
            ["Integrated AI/ML Systems"],
            ["Deployed Production Infrastructure"]
        ]
        
        expertise_table = Table(expertise_data, colWidths=[8*inch])
        expertise_table.setStyle(TableStyle([
            ('FONT', (0, 0), (-1, -1), 'Helvetica', 16),
            ('TEXTCOLOR', (0, 0), (-1, -1), DEXTER_DARK),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 15),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
            ('LINEBELOW', (0, 0), (-1, -2), 1, DEXTER_LIGHT_PURPLE),
        ]))
        
        self.story.append(expertise_table)
        self.story.append(PageBreak())
        
    def add_contact_slide(self):
        self.story.append(Spacer(1, 2*inch))
        self.story.append(Paragraph("Contact", self.styles['SectionHeader']))
        self.story.append(Spacer(1, 0.5*inch))
        
        contact_info = [
            "Email: meltedmindz1@gmail.com",
            "Twitter: @Dexter_AI_",
            "GitHub: MeltedMindz/Dexter",
            "Website: dexteragent.com"
        ]
        
        for info in contact_info:
            self.story.append(Paragraph(info, self.styles['DexterBody']))
            
    def build(self):
        # Add all slides
        self.add_title_slide()
        self.add_problem_slide()
        self.add_solution_slide()
        self.add_market_opportunity_slide()
        self.add_competitive_advantages_slide()
        self.add_roadmap_slide()
        self.add_funding_slide()
        self.add_team_slide()
        self.add_contact_slide()
        
        # Build the PDF
        self.doc.build(self.story, onFirstPage=self._add_page_number, onLaterPages=self._add_page_number)
        print(f"PDF deck created: {self.filename}")
        
    def _add_page_number(self, canvas, doc):
        canvas.saveState()
        canvas.setFont('Helvetica', 10)
        canvas.setFillColor(DEXTER_GRAY)
        page_num = canvas.getPageNumber()
        text = f"Dexter Protocol | {page_num}"
        canvas.drawRightString(self.width - 0.75*inch, 0.5*inch, text)
        canvas.restoreState()

if __name__ == "__main__":
    deck = DexterPitchDeck()
    deck.build()