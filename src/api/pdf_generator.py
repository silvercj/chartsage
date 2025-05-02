from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.units import inch
import plotly.graph_objects as go
from typing import List, Dict, Any
import tempfile
import os

class PDFGenerator:
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.doc = SimpleDocTemplate(
            output_path,
            pagesize=landscape(letter),
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        self.styles = getSampleStyleSheet()
        self._setup_styles()

    def _setup_styles(self):
        """Set up custom styles for the PDF."""
        self.styles.add(ParagraphStyle(
            name='Title',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30
        ))
        self.styles.add(ParagraphStyle(
            name='Heading2',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=12
        ))
        self.styles.add(ParagraphStyle(
            name='Description',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=12
        ))

    def generate_pdf(self, figures: List[go.Figure], specs: List[Dict[str, Any]], title: str = "Data Insights"):
        """Generate a PDF with visualizations and descriptions."""
        story = []
        
        # Add title
        story.append(Paragraph(title, self.styles['Title']))
        story.append(Spacer(1, 12))

        # Create a temporary directory for the images
        with tempfile.TemporaryDirectory() as temp_dir:
            # Add each visualization and its description
            for i, (fig, spec) in enumerate(zip(figures, specs)):
                # Add section title
                story.append(Paragraph(f"{i+1}. {spec['title']}", self.styles['Heading2']))
                story.append(Spacer(1, 12))
                
                # Add description
                story.append(Paragraph(spec['description'], self.styles['Description']))
                story.append(Spacer(1, 12))
                
                # Save the figure as a temporary image
                img_path = os.path.join(temp_dir, f"viz_{i}.png")
                fig.write_image(img_path, scale=2, width=900, height=500)
                
                # Add the image to the PDF
                img = Image(img_path)
                img.drawHeight = 4 * inch
                img.drawWidth = 8 * inch
                story.append(img)
                
                # Add some space after the image
                story.append(Spacer(1, 24))
                
                # Add a page break after each visualization except the last one
                if i < len(figures) - 1:
                    story.append(PageBreak())

        # Build the PDF
        self.doc.build(story)

    def add_metadata(self, metadata: Dict[str, str]):
        """Add metadata to the PDF."""
        self.doc.title = metadata.get('title', '')
        self.doc.author = metadata.get('author', '')
        self.doc.subject = metadata.get('subject', '')
        self.doc.creator = metadata.get('creator', 'ChartSage') 