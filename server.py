from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import re
import os
import openai
from math import ceil
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import pypdf
import torch
import torch.nn as nn
import pdfplumber
from dotenv import load_dotenv
load_dotenv()


app = Flask(__name__)
CORS(app)  

import json
import re
import os
import openai
from math import ceil



class DoctorLLM:
    def __init__(self, lab_data, model="gpt-4o", textbook_path="chat_api/output_full_intmed2.txt", api_key=None):
        # Set up OpenAI API
        self.model = model
        self.lab_data = lab_data
        if api_key:
            openai.api_key = api_key
        else:
            # Try to get API key from environment variable
            openai.api_key = os.environ.get("OPENAI_API_KEY")
            if not openai.api_key:
                raise ValueError("No OpenAI API key provided. Please set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        # Initialize conversation tracking
        self.conversation_history = []
        self.question_count = 0
        self.max_questions = 8  # Maximum number of questions to ask
        self.patient_info = {
            "symptoms": [],
            "duration": {},
            "severity": {},
            "medical_history": [],
            "medications": [],
            "allergies": []
        }
        self.diagnosis = ""
        self.treatment_plan = ""
        self.referral = ""
        
        # Load and process the medical textbook
        self.textbook_chunks = self.load_textbook(textbook_path)
        self.num_chunks = len(self.textbook_chunks)
        print(f"Medical textbook loaded ({self.num_chunks} chunks)")
        print("[Medical Consultation Started]")
        print("Type 'exit' to end the consultation.")

    def load_textbook(self, textbook_path, chunk_size=5000, overlap=500):
        """Load the medical textbook and split it into manageable chunks"""
        try:
            with open(textbook_path, 'r', encoding='utf-8') as file:
                textbook_content = file.read()
            
            # Split the textbook into chunks with overlap
            chunks = []
            content_length = len(textbook_content)
            
            for i in range(0, content_length, chunk_size - overlap):
                chunk_end = min(i + chunk_size, content_length)
                chunk = textbook_content[i:chunk_end]
                chunks.append(chunk)
            
            print(f"Successfully loaded medical textbook from {textbook_path}")
            return chunks
        except Exception as e:
            print(f"Error loading textbook: {e}")
            print("Proceeding without textbook knowledge")
            return ["No medical textbook data available."]

    def retrieve_relevant_chunks(self, query, top_k=3):
        """Find the most relevant chunks from the textbook for a given query"""
        # Create a simplified representation of all chunks for matching
        chunks_overview = []
        for i, chunk in enumerate(self.textbook_chunks):
            # Get first 100 chars of chunk as preview
            preview = chunk[:100].replace('\n', ' ').strip()
            chunks_overview.append(f"Chunk {i}: {preview}...")
        
        # Select a random sample of chunks if there are too many
        max_chunks_to_sample = 20
        if len(chunks_overview) > max_chunks_to_sample:
            import random
            sampled_indices = sorted(random.sample(range(len(chunks_overview)), max_chunks_to_sample))
            sample_text = "\n".join([chunks_overview[i] for i in sampled_indices])
        else:
            sampled_indices = list(range(len(chunks_overview)))
            sample_text = "\n".join(chunks_overview)
        
        # Find relevant chunks using OpenAI
        system_prompt = "You are a helpful medical assistant that helps find relevant medical information."
        user_prompt = f"""
        Given this medical query: "{query}"
        
        Find the most relevant chunks from these medical textbook previews:
        {sample_text}
        
        Return only the indices of the {min(top_k, len(sampled_indices))} most relevant chunks.
        Format your response as a comma-separated list of numbers only.
        """
        
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0
            )
            result = response.choices[0].message.content
            
            # Extract numbers from response
            number_list = re.findall(r'\d+', result)
            chunk_indices = []
            
            for num_str in number_list:
                try:
                    # Convert to integer and check if it's a valid index
                    idx = int(num_str)
                    if idx < len(self.textbook_chunks):
                        if sampled_indices:
                            # Map back to original index if we sampled
                            if idx < len(sampled_indices):
                                chunk_indices.append(sampled_indices[idx])
                        else:
                            chunk_indices.append(idx)
                except ValueError:
                    continue
                    
            # Limit to top_k unique indices
            unique_indices = []
            for idx in chunk_indices:
                if idx not in unique_indices and len(unique_indices) < top_k:
                    unique_indices.append(idx)
                    
            # If we didn't get enough chunks, add some sequential ones
            if not unique_indices:
                unique_indices = list(range(min(top_k, len(self.textbook_chunks))))
                
            # Return the actual text chunks
            return [self.textbook_chunks[i] for i in unique_indices]
            
        except Exception as e:
            print(f"Error retrieving chunks: {e}")
            # Fallback to first few chunks
            return self.textbook_chunks[:min(top_k, len(self.textbook_chunks))]

    def generate_response(self, prompt, include_context=True):
        """Generate a response using OpenAI with medical textbook context"""
        
        # Format conversation history for context
        formatted_history = []
        for i, msg in enumerate(self.conversation_history):
            role = "Doctor" if i % 2 == 0 else "Patient"
            formatted_history.append(f"{role}: {msg}")
        
        context = "\n".join(formatted_history)
        
        # Safely format patient info to avoid type errors
        symptoms_str = ', '.join(self.safe_list_to_strings(self.patient_info['symptoms']))
        duration_str = ', '.join([f"{k}: {v}" for k, v in self.patient_info['duration'].items()]) if isinstance(self.patient_info['duration'], dict) else "Not specified"
        medical_history_str = ', '.join(self.safe_list_to_strings(self.patient_info['medical_history']))
        
        # Summarize patient info for context
        patient_summary = f"""
Patient Information:
- Reported symptoms: {symptoms_str if symptoms_str else 'None reported yet'}
- Duration: {duration_str if duration_str else 'Not specified'}
- Medical history: {medical_history_str if medical_history_str else 'None reported'}
- Current question count: {self.question_count} of {self.max_questions}
- Lab Data: {self.lab_data}
"""

        # Build the system prompt
        system_prompt = """
You are an experienced and compassionate doctor conducting a patient consultation. 
Your goal is to gather relevant information and provide helpful medical guidance.

Important guidelines:
1. Use simple, easy-to-understand language (avoid medical jargon when possible)
2. Ask only ONE question at a time about a single topic 
3. Be concise but friendly and compassionate
4. Don't repeat questions that have already been answered
5. Track the question count to complete the consultation in 6-8 questions total
6. Acknowledge information the patient has already shared
7. Consider the patient's lab test results in your assessment when available
"""

        # Build the user message content
        user_content = f"""
CONVERSATION HISTORY:
{context}

PATIENT SUMMARY:
{patient_summary}
"""

        # Add medical textbook context if needed and available
        if include_context and len(self.textbook_chunks) > 1:
            # Get relevant chunks based on patient info and prompt
            symptom_str = symptoms_str if symptoms_str else "unspecified symptoms"
            query = f"Patient with {symptom_str}. {prompt}"
            relevant_chunks = self.retrieve_relevant_chunks(query)
            
            # Limit textbook content to reasonable size
            textbook_content = ""
            total_length = 0
            for chunk in relevant_chunks:
                if total_length + len(chunk) > 6000:  # Set a reasonable limit
                    textbook_content += chunk[:6000-total_length] + "..."
                    break
                textbook_content += chunk + "\n\n"
                total_length += len(chunk) + 2
                
            user_content += f"""
MEDICAL REFERENCE INFORMATION:
{'-' * 40}
{textbook_content}
{'-' * 40}
"""

        user_content += f"""
TASK:
{prompt}
"""
        
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating response: {str(e)[:100]}...")
            return "I'm having some technical difficulties. Let's continue our discussion."
    
    def safe_list_to_strings(self, items):
        """Safely convert any list of items to a list of strings"""
        result = []
        if isinstance(items, list):
            for item in items:
                if isinstance(item, (str, int, float)):
                    result.append(str(item))
                elif isinstance(item, dict):
                    # Convert dict to string representation
                    result.append(str(item))
                # Ignore other types
        return result


    def extract_medical_info(self, user_input):
        """Improved medical information extraction"""
        # Skip empty inputs
        if not user_input.strip():
            return self.patient_info
        
        # Create a more detailed prompt
        system_prompt = """
        You are a medical information extraction system. Extract ALL medical information from patient statements.
        Return JSON with these keys (fill ALL keys even if empty):
        - symptoms: [list]
        - duration: {"symptom": "duration"} 
        - severity: {"symptom": "severity"}
        - medical_history: [list]
        - medications: [list]
        - allergies: [list]
        """
        
        user_prompt = f"""
        PATIENT STATEMENT: "{user_input}"
        
        Extract in this exact JSON format (include ALL keys):
        {{
            "symptoms": [],
            "duration": {{}},
            "severity": {{}},
            "medical_history": [],
            "medications": [],
            "allergies": []
        }}
        """
        
        try:
            response = openai.chat.completions.create(
                model="gpt-4o",  # More reliable for JSON
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            extracted = json.loads(response.choices[0].message.content)
            
            # Merge new data with existing
            for key in ['symptoms', 'medical_history', 'medications', 'allergies']:
                if isinstance(extracted.get(key, None), list):
                    for item in extracted[key]:
                        if item and item not in self.patient_info[key]:
                            self.patient_info[key].append(str(item))
            
            # Handle dictionaries
            for key in ['duration', 'severity']:
                if isinstance(extracted.get(key, None), dict):
                    for k, v in extracted[key].items():
                        if k and v:
                            self.patient_info[key][str(k)] = str(v)
                            
        except Exception as e:
            print(f"Extraction error: {str(e)[:200]}")
        
        return self.patient_info
    
    def get_next_question(self):
        """Generate the next question based on current consultation progress"""
        prompt = ""
        
        # Increment question count
        self.question_count += 1
        
        # Check if we've reached the max questions
        if self.question_count >= self.max_questions:
            return self.provide_assessment()
        
        # Generate appropriate question based on question count and collected information
        prompt = f"""
The patient consultation is at question {self.question_count} of {self.max_questions}.

Based on the information collected so far, generate ONE simple, clear follow-up question that will help with diagnosis.

Focus on: 
- Information not yet collected that would be most helpful at this stage
- Ask about only ONE aspect at a time (not multiple questions)
- Use simple language that patients can easily understand
- If you already have enough symptom information, ask about medical history, medications, or lifestyle factors

Remember: We need to complete the consultation within {self.max_questions} total questions, and we're currently at question {self.question_count}.
"""
        
        return self.generate_response(prompt)
    
    def provide_assessment(self):
        """Generate a medical assessment based on collected information"""
        # Safely format patient info
        symptoms_str = ', '.join(self.safe_list_to_strings(self.patient_info['symptoms']))
        
        # Handle duration and severity safely
        duration_items = []
        if isinstance(self.patient_info['duration'], dict):
            for k, v in self.patient_info['duration'].items():
                duration_items.append(f"{k}: {v}")
        duration_str = ', '.join(duration_items) if duration_items else 'Not specified'
        
        severity_items = []
        if isinstance(self.patient_info['severity'], dict):
            for k, v in self.patient_info['severity'].items():
                severity_items.append(f"{k}: {v}")
        severity_str = ', '.join(severity_items) if severity_items else 'Not specified'
        
        # Handle lists safely
        medical_history_str = ', '.join(self.safe_list_to_strings(self.patient_info['medical_history']))
        medications_str = ', '.join(self.safe_list_to_strings(self.patient_info['medications']))
        allergies_str = ', '.join(self.safe_list_to_strings(self.patient_info['allergies']))
        
        prompt = f"""
Based on the patient information collected, provide a clear summary and plan:

Reported symptoms: {symptoms_str if symptoms_str else 'None reported'}
Duration information: {duration_str}
Severity information: {severity_str}
Medical history: {medical_history_str if medical_history_str else 'None reported'}
Current medications: {medications_str if medications_str else 'None reported'}
Allergies: {allergies_str if allergies_str else 'None reported'}

Provide a conclusion with these FOUR sections:
1. "Your Symptoms": Summarize the key symptoms in simple terms
2. "Likely Diagnosis": Provide the most likely condition(s) based on symptoms
3. "Treatment Plan": Suggest specific next steps, including any medications, tests, or lifestyle changes
4. "Whom to refer" : Suggest which type of specialist doctor to further refer 
5. Key Insights, Recommendations and Risk Factors

Use simple, clear language. Avoid medical jargon when possible.
"""
        
        response = self.generate_response(prompt)
        
        # Extract diagnosis, treatment plan, and referral from response
        diagnosis_match = re.search(r'Likely Diagnosis:?(.*?)(?:Treatment Plan:|$)', response, re.DOTALL)
        if diagnosis_match:
            self.diagnosis = diagnosis_match.group(1).strip()
            
        treatment_match = re.search(r'Treatment Plan:?(.*?)(?:Whom to refer:|$)', response, re.DOTALL)
        if treatment_match:
            self.treatment_plan = treatment_match.group(1).strip()
            
        referral_match = re.search(r'Whom to refer:?(.*?)$', response, re.DOTALL)
        if referral_match:
            self.referral = referral_match.group(1).strip()

        sections = {
    "symptoms": r"(?:Your Symptoms|Symptoms):?\s*(.*?)(?:\n\n|\Z)",
    "diagnosis": r"(?:Likely Diagnosis|Diagnosis):?\s*(.*?)(?:\n\n|\Z)",
    "treatment": r"(?:Treatment Plan|Treatment):?\s*(.*?)(?:\n\n|\Z)", 
    "referral": r"(?:Whom to refer|Referral):?\s*(.*?)(?:\n\n|\Z)"
}

        for key, pattern in sections.items():
            match = re.search(pattern, response, re.DOTALL|re.IGNORECASE)
            if match:
                setattr(self, key, match.group(1).strip())    
            
        # Save the assessment to a JSON file
        self.save_assessment_to_json()
            
        return response
    

    def process_user_input(self, user_input):
        """Process user input and generate appropriate response"""
        # Add to conversation history
        self.conversation_history.append(user_input)
        
        # Extract medical information from input
        self.extract_medical_info(user_input)
        
        # Generate response based on consultation progress
        response = self.get_next_question()
        
        # Add response to conversation history
        self.conversation_history.append(response)
        
        return response

    def save_assessment_to_json(self):
        """Save complete consultation data"""
        assessment_data = {
            "patient_info": self.patient_info,
            "diagnosis": self.diagnosis,
            "treatment_plan": self.treatment_plan,
            "referral": self.referral,
            "consultation_stats": {
                "questions_asked": self.question_count,
                "max_questions": self.max_questions
            },
            "conversation_history": self.conversation_history[-10:],  # Last 10 exchanges
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"assessment_{timestamp}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(assessment_data, f, indent=2, ensure_ascii=False)
                
            print(f"✅ Saved complete assessment to {filename}")
        except Exception as e:
            print(f"❌ Failed to save assessment: {str(e)[:100]}")

    def chat(self):
        """Run the chat interface"""
            # Initial greeting
        initial_greeting = "Hi there! I'm your virtual doctor today. What brings you in?"
        print(f"Doctor: {initial_greeting}")
        self.conversation_history.append(initial_greeting)
            
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit", "bye"]:
                closing_statement = "Thank you for talking with me today. Take care, and remember to seek in-person medical care if your symptoms get worse."
                print(f"Doctor: {closing_statement}")
                break
                
            doctor_response = self.process_user_input(user_input)
            print(f"Doctor: {doctor_response}")
                
                # Check if we've reached the assessment stage
            if self.question_count >= self.max_questions:
                print("\nConsultation complete. Assessment saved to JSON file.")
                break


def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    
def extract_lab_tests_dict(response_text):
    pattern = r"[-•]?\s*([\w\s/()%.-]+?):\s*([\d.]+)\s*(\w+/?.*)?"
    matches = re.findall(pattern, response_text)
    lab_dict = {}
    for test, value, unit in matches:
        test = test.strip()
        try:
            lab_dict[test] = float(value)
        except ValueError:
            continue
    return lab_dict    

def print_extracted_lab_values(pdf_path):
    # Step 1: Extract text from PDF
    text = extract_text_from_pdf(pdf_path)

    # Step 2: Chunk and embed
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.create_documents([text])

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = FAISS.from_documents(docs, embedding=embedding_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="refine")

    # Step 3: Ask for lab test values
    query = "List lab test names and values only with units (no suggestions). Format: Test: Value Unit"
    lab_tests_response = qa.run(query)

    # Step 4: Extract lab values
    lab_data = extract_lab_tests_dict(lab_tests_response)

    return lab_data


doctor_instance = None


@app.route('/health', methods=['GET'])
def health():
    print('evwev')
    return jsonify({
            'status': 'success'
        }) 

@app.route('/start_consultation', methods=['POST'])
def start_consultation():
    global doctor_instance

    model = request.form.get('model', 'gpt-4o')
    textbook_path = request.form.get('textbook_path', 'doc_final_api/output_full_intmed2.txt')
    api_key = request.form.get('api_key', os.environ.get("OPENAI_API_KEY"))
    print("Model:", model)
    print("Textbook Path:", textbook_path)
    print("API Key:", api_key)

    # Handle file upload
    os.makedirs('uploads', exist_ok=True)
    file = request.files['file']
    file.save(os.path.join('uploads', file.filename))
    print("File saved:", file.filename)

    lab_data = print_extracted_lab_values(file.filename)
    # return 'File uploaded successfully'

    
    try:
        # Initialize doctor instance
        doctor_instance = DoctorLLM(
            lab_data=lab_data,
            model=model,
            textbook_path=textbook_path,
            api_key=api_key
        )
        
        # Get initial greeting
        initial_greeting = "Hi there! I'm your virtual doctor today. What brings you in?"
        doctor_instance.conversation_history.append(initial_greeting)
        
        return jsonify({
            'status': 'success',
            'message': initial_greeting,
            'question_count': doctor_instance.question_count,
            'max_questions': doctor_instance.max_questions
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to start consultation: {str(e)}'
        }), 400

@app.route('/process_input', methods=['POST'])
def process_input():
    global doctor_instance
    
    if not doctor_instance:
        return jsonify({
            'status': 'error',
            'message': 'Consultation not started. Please call /start_consultation first.'
        }), 400
    
    data = request.get_json()
    user_input = data.get('input', '')
    
    if not user_input:
        return jsonify({
            'status': 'error',
            'message': 'No input provided'
        }), 400
    
    try:
        # Process user input
        response = doctor_instance.process_user_input(user_input)
        
        # Check if consultation is complete
        consultation_complete = doctor_instance.question_count >= doctor_instance.max_questions
        
        return jsonify({
            'status': 'success',
            'response': response,
            'question_count': doctor_instance.question_count,
            'max_questions': doctor_instance.max_questions,
            'consultation_complete': consultation_complete,
            'patient_info': doctor_instance.patient_info,
            'diagnosis': doctor_instance.diagnosis if consultation_complete else None,
            'treatment_plan': doctor_instance.treatment_plan if consultation_complete else None,
            'referral': doctor_instance.referral if consultation_complete else None
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to process input: {str(e)}'
        }), 500

@app.route('/end_consultation', methods=['POST'])
def end_consultation():
    global doctor_instance
    
    if not doctor_instance:
        return jsonify({
            'status': 'error',
            'message': 'No active consultation to end'
        }), 400
    
    try:
        # Generate final assessment if not already done
        if doctor_instance.question_count < doctor_instance.max_questions:
            response = doctor_instance.provide_assessment()
        else:
            response = "Consultation completed. Here's your assessment again."
        
        # Get the saved assessment data
        assessment_data = {
            'symptoms': doctor_instance.patient_info["symptoms"],
            'duration': doctor_instance.patient_info["duration"],
            'severity': doctor_instance.patient_info["severity"],
            'medical_history': doctor_instance.patient_info["medical_history"],
            'medications': doctor_instance.patient_info["medications"],
            'allergies': doctor_instance.patient_info["allergies"],
            'diagnosis': doctor_instance.diagnosis,
            'treatment_plan': doctor_instance.treatment_plan,
            'referral': doctor_instance.referral,
            'consultation_questions': doctor_instance.question_count
        }
        
        # Clear the doctor instance
        doctor_instance = None
        
        return jsonify({
            'status': 'success',
            'message': response,
            'assessment': assessment_data
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to end consultation: {str(e)}'
        }), 500

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5001, debug=True)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)  # Render uses dynamic $PORT
