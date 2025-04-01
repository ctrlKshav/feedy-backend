
from fastapi import FastAPI, Form, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import cloudinary
import cloudinary.uploader
import os
from typing import List

from google import genai
from google.genai import types
import requests

from pydantic import BaseModel
from typing import List

class ImageInfo(BaseModel):
    image_url: str
    image_name: str

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
        # Only run this block for Gemini Developer API
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        response = client.models.generate_content(
            model='gemini-2.0-pro-exp-02-05', 
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=500,
                temperature=0.8),
            contents=combined_prompt
            )
        refined_prompt = response.text


        # completion = client.chat.completions.create(
        #     model="llama-3.2-90b-vision-preview",
        #     messages=[
        #         {"role": "system", "content": system_prompt},
        #         {"role": "user", "content": combined_prompt},
        #     ],
        #     temperature=0.85,
        #     max_completion_tokens=1024,
        #     top_p=1,
        #     stream=False,
        # )

        # if not completion.choices:
        #     raise HTTPException(status_code=500, detail="AI response was empty.")

        # refined_prompt = completion.choices[0].message.content

        return JSONResponse(content={"refined_prompt": refined_prompt, "status": "success"})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "status": "error"})


@app.post("/upload-images")
async def upload_images(images: list[UploadFile] = File(...)):
    try:
        uploaded_images = []
        for image in images:
            if not image.filename.lower().endswith(("png", "jpg", "jpeg", "webp")):
                raise HTTPException(status_code=400, detail="Unsupported file format. Please upload PNG or JPG images.")

            # Upload file to Cloudinary
            result = cloudinary.uploader.upload(image.file)
            image_url = result.get("url")
            
            if not image_url:
                raise HTTPException(status_code=500, detail="Image upload failed.")
                
            uploaded_images.append({
                "image_url": image_url,
                "image_name": image.filename.lower()
            })
        
        return JSONResponse(content={"images": uploaded_images, "status": "success"})

    except HTTPException as http_err:
        return JSONResponse(status_code=http_err.status_code, content={"error": http_err.detail, "status": "error"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Internal server error: {str(e)}", "status": "error"})

@app.post("/analyze-images")
async def analyze_images(request: AnalysisRequest):
    print("hel2")
    print(request.admin_persona)
    try:
        analysis = []
        combined_prompt = f"""Adopt this professional persona:
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
            completion = client.chat.completions.create(
                model="llama-3.2-11b-vision-preview",
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
            })
        
        return JSONResponse(content=analysis)
    except Exception as e:
        return JSONResponse(status_code=500, content={"response": f"Internal server error: {str(e)}", "status": "error"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)