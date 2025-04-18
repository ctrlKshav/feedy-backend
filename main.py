from fastapi import FastAPI, Form, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from dotenv import load_dotenv
import cloudinary
import cloudinary.uploader
import os
import base64
import io
import PyPDF2
from typing import List

from pydantic import BaseModel
from typing import List

class ImageInfo(BaseModel):
    image_url: str
    image_name: str
    file_type: str = "image"  # Default to "image", can be "pdf" for PDF files
    pdf_text: str = None  # Text content for PDF files

class AnalysisRequest(BaseModel):
    image_urls: List[ImageInfo]
    question: str
    admin_persona: str

# Define request model for the new endpoint
class RefinePersonaRequest(BaseModel):
    initial_prompt: str


app = FastAPI()
load_dotenv()

# Configure Cloudinary
cloudinary.config(  
    cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME'),
    api_key=os.getenv('CLOUDINARY_API_KEY'),
    api_secret=os.getenv('CLOUDINARY_API_SECRET')
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq client
client = Groq(api_key=os.getenv('GROQ_API_KEY'))

# Persona description
admin_persona = """You are an experienced UX Design Manager with over 15 years of experience in leading design teams at top tech companies. Your feedback approach:
ANALYSIS:
- Evaluate visual hierarchy and information architecture
- Assess accessibility compliance (WCAG guidelines)
- Review consistency with design systems
- Analyze user flow and interaction patterns
FEEDBACK STYLE:
- Start with positive aspects before addressing areas for improvement
- Provide specific, actionable recommendations
- Reference UX best practices and research data
- Consider business goals and user needs equally
- Use the "feedback sandwich" method
KEY FOCUS AREAS:
1. Usability:
   - Clarity of navigation
   - Ease of interaction
   - Error prevention
   - User feedback mechanisms
2. Visual Design:
   - Color contrast and accessibility
   - Typography hierarchy
   - Spacing and layout
   - Visual consistency
3. User Flow:
   - Task completion efficiency
   - Number of steps
   - Clear call-to-actions
   - Error recovery paths
4. Business Impact:
   - Conversion optimization
   - User engagement
   - Brand alignment
   - Scalability
DELIVERY GUIDELINES:
- Be constructive and specific
- Provide examples and references
- Suggest A/B testing opportunities
- Include metrics for success measurement
"""

@app.post("/refine-persona")
async def refine_persona(request: RefinePersonaRequest):
    try:
        system_prompt = """You are an expert AI persona architect. Transform basic descriptions into polished, structured personas with:
        1. Authentic personality mirroring the input tone
        2. Detailed operational frameworks
        3. Practical design industry relevance
        4. Scenario-based examples
        Maintain all key traits from the input while adding professional structure."""

        structure_guide = """**Refined Persona Structure**
        
## Persona Overview
- Name (create if missing)
- Role/Title
- Experience Level
- Key Style Adjectives
- Core Philosophy

## Core Competencies
- Design Specializations
- Technical Proficiencies
- Methodology Preferences

## Interaction Framework
- Communication Tone
- Feedback Approach
- Questioning Style
- Conflict Resolution

## Visual Identity Guidelines
- Color Palette Preferences
- Layout Principles
- Typography Standards
- Accessibility Standards

## Example Scenarios
1. [Client Type]: [Challenge] -> [Solution Approach]
2. [Client Type]: [Challenge] -> [Solution Approach]"""

        combined_prompt = f"""
        Input Persona: {request.initial_prompt}

        Transform this into a professional designer persona using:
        {structure_guide}

        """


        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": combined_prompt},
            ],
            temperature=0.85,
            max_completion_tokens=1024,
            top_p=1,
            stream=False,
        )

        if not completion.choices:
            raise HTTPException(status_code=500, detail="AI response was empty.")

        refined_prompt = completion.choices[0].message.content

        return JSONResponse(content={"refined_prompt": refined_prompt, "status": "success"})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "status": "error"})


@app.post("/upload-images")
async def upload_images(images: list[UploadFile] = File(...)):
    try:
        uploaded_images = []
        for image in images:
            file_extension = os.path.splitext(image.filename.lower())[1]
            is_pdf = file_extension == '.pdf'
            
            # Validate file format
            allowed_extensions = ['.png', '.jpg', '.jpeg', '.webp', '.pdf']
            if file_extension not in allowed_extensions:
                raise HTTPException(status_code=400, detail=f"Unsupported file format. Please upload PNG, JPG, or PDF files. Got: {file_extension}")
            
            # For PDFs, we'll extract text and store it separately
            pdf_text = None
            if is_pdf:
                try:
                    # Read PDF content and extract text
                    pdf_content = await image.read()
                    pdf_file = io.BytesIO(pdf_content)
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    
                    # Extract text from all pages
                    pdf_text = ""
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        pdf_text += page.extract_text() + "\n\n"
                    
                    # Reset file cursor for Cloudinary upload
                    await image.seek(0)
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"PDF text extraction failed: {str(e)}")
            
            # Upload file to Cloudinary
            result = cloudinary.uploader.upload(image.file)
            image_url = result.get("url")
            
            if not image_url:
                raise HTTPException(status_code=500, detail="File upload failed.")
            
            # Determine file type
            file_type = "pdf" if is_pdf else "image"
            
            # Prepare response object
            image_info = {
                "image_url": image_url,
                "image_name": image.filename.lower(),
                "file_type": file_type
            }
            
            # Add pdf_text if available
            if pdf_text:
                image_info["pdf_text"] = pdf_text
                
            uploaded_images.append(image_info)
        
        return JSONResponse(content={"images": uploaded_images, "status": "success"})

    except HTTPException as http_err:
        return JSONResponse(status_code=http_err.status_code, content={"error": http_err.detail, "status": "error"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Internal server error: {str(e)}", "status": "error"})

@app.post("/analyze-images")
async def analyze_images(request: AnalysisRequest):
    try:
        analysis = []
        base_prompt = f"""Adopt this professional persona:
        {request.admin_persona if request.admin_persona else admin_persona}

        Using your expertise, conduct a thorough analysis of the provided design. Follow this structure:

            **ANALYSIS FRAMEWORK**
            1. **First Impressions**
            - Share your immediate professional assessment
            - Brand alignment evaluation
            - Functional clarity assessment

            2. **Detailed Evaluation** (Use bullet points)
            [✔] **Strengths**:
            {{{{bullet points highlighting exemplary elements}}}}
            
            [⚠️] **Opportunities**:
            {{{{bullet points proposing targeted improvements}}}}

            3. **Professional Recommendations**
            - Critical revisions (urgent needs)
            - Value-add refinements (strategic improvements)
            - Testing opportunities (proven optimization approaches)

            4. **Expert Considerations**
            - Accessibility audit (WCAG 2.1+ compliance)
            - Responsive design integrity
            - Cross-platform performance
            - Cognitive ergonomics

            **FORMATTING REQUIREMENTS**
            - Maintain authoritative yet collaborative tone
            - Cite relevant design methodologies from your expertise
            - Flag implementation effort (Low/Medium/High)
            - Use markdown bolding for section headers
            

            **Specific Focus**: {request.question}

            IMPORTANT: Present as first-person expert analysis using "I recommend"/"My assessment shows". Never qualify statements with AI references. Fully own your professional perspective."""
            
        for image in request.image_urls:
            # Determine if this is a PDF file
            is_pdf = image.file_type == "pdf"
            
            if is_pdf:
                # For PDFs, use a text-only model with extracted text
                pdf_prompt = f"""Adopt this professional persona:
                {request.admin_persona if request.admin_persona else admin_persona}

                I'm going to provide you with text extracted from a PDF document. Please analyze this as a document design expert, focusing on:
                
                - Document structure assessment
                - Information architecture and hierarchy
                - Typography and readability
                - Content organization
                - Clarity and effectiveness of communication
                
                Using your expertise, conduct a thorough analysis of the provided document. Follow this structure:

                **ANALYSIS FRAMEWORK**
                1. **First Impressions**
                - Share your professional assessment of the document
                - Purpose clarity assessment
                - Overall effectiveness evaluation

                2. **Detailed Evaluation** (Use bullet points)
                [✔] **Strengths**:
                {{{{bullet points highlighting exemplary elements}}}}
                
                [⚠️] **Opportunities**:
                {{{{bullet points proposing targeted improvements}}}}

                3. **Professional Recommendations**
                - Critical revisions (urgent needs)
                - Value-add refinements (strategic improvements)
                - Testing opportunities (proven optimization approaches)

                **FORMATTING REQUIREMENTS**
                - Maintain authoritative yet collaborative tone
                - Cite relevant document design methodologies from your expertise
                - Flag implementation effort (Low/Medium/High)
                - Use markdown bolding for section headers
                
                **Specific Focus**: {request.question}

                IMPORTANT: Present as first-person expert analysis using "I recommend"/"My assessment shows". Never qualify statements with AI references. Fully own your professional perspective.
                
                Here's the extracted text from the PDF:
                
                {image.pdf_text}
                """
                
                # Use text-only model for PDF analysis
                completion = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",  # Using text-only model 
                    messages=[
                        {
                            "role": "user",
                            "content": pdf_prompt
                        }
                    ],
                    temperature=0.7,
                    max_completion_tokens=1024,
                    top_p=1,
                    stream=False
                )
            else:
                # For images, use the vision-capable model
                combined_prompt = base_prompt
                
                completion = client.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",  # Using vision-capable model
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": combined_prompt},
                                {"type": "image_url", "image_url": {"url": image.image_url}}
                            ]
                        }
                    ],
                    temperature=0.7,
                    max_completion_tokens=1024,
                    top_p=1,
                    stream=False
                )

            if not completion.choices:
                raise HTTPException(status_code=500, detail="AI response was empty.")

            analysis.append({
                "response": completion.choices[0].message.content,
                "status": "success",
                "image_name": image.image_name,
                "image_url": image.image_url,
                "file_type": image.file_type
            })
        
        return JSONResponse(content=analysis)
    except Exception as e:
        return JSONResponse(status_code=500, content={"response": f"Internal server error: {str(e)}", "status": "error"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)