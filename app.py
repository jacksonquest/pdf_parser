import streamlit as st
import fitz
from pdf2image import convert_from_bytes
import json
from together import Together
import base64
import io
import pandas as pd

# Initialize Together client
client = Together(api_key=st.secrets["TOGETHER_API_KEY"])

def encode_image(image):
    """Convert image to base64 encoding."""
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    return base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

def process_image_with_llama(image):
    """Send image to Llama 3.2 Vision via Together API."""
    base64_image = encode_image(image)
    
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract information about Germany, Revenue, and EBITDA from this image:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                ],
            }
        ],
        stream=False,
    )
    
    if response and hasattr(response, 'choices') and response.choices:
        text_response = response.choices[0].message.content
        return text_response
    else:
        return None
    
def extract_text_from_pdf(uploaded_file):
    """Extract text from PDF and return a dictionary of pages containing 'Germany'."""
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    pages_with_germany = {}
    for page_num in range(len(doc)):
        text = doc[page_num].get_text("text")
        if "Germany" in text:
            pages_with_germany[page_num] = text
    return pages_with_germany


def structure_extracted_data(extracted_data):
    """Pass extracted data to Llama 3.2 Vision to structure it into a tabular format."""
    prompt = """
    You are an expert data extraction specialist. Your task is to process the provided text snippets, each representing a page from a document, and extract key financial information related to Germany, specifically Revenue and EBITDA (and related metrics like Service Revenue, Broadband Revenue, etc.).
    
    Return the extracted data strictly as a **list of JSON objects**, where each object represents a row in a table with the following keys:
    
    [
        {
            "Page Number": 1,
            "Financial Metric": "Revenue",
            "Time Period Start": "2024-01-01",
            "Time Period End": "2024-12-31",
            "Value Type": "Actual",
            "Numeric Value": 5000000,
            "Unit": "EUR",
            "Growth Value": 200000,
            "Growth Unit": "EUR",
            "Growth Direction": "Up",
            "Lower Bound": null,
            "Upper Bound": null,
            "Specific Context": "Annual revenue for Germany"
        }
    ]
    
    If any information is missing, set the value as `null`. Do not include explanations or extra text. Only return a valid JSON list.
    """
    
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        max_tokens=5000,
        messages=[
            {"role": "user", "content": prompt + json.dumps(extracted_data, indent=4)}
        ],
        stream=False,
    )
    
    if response and hasattr(response, 'choices') and response.choices:
        structured_data = response.choices[0].message.content
        return structured_data
    else:
        return None

def main():
    st.set_page_config(layout="wide")

    st.markdown("""
            <h1 style="text-align: center;">PDF Parser (powered by LLM)</h1>
        """, unsafe_allow_html=True)
    
    if "df" not in st.session_state:
        st.session_state["df"] = None

    if "uploaded_file" not in st.session_state:
        st.session_state["uploaded_file"] = None

    # Step 1: Upload PDF
    st.session_state.uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if st.session_state.uploaded_file:
        st.write("Processing PDF...")
        
        # Step 2: Extract text and filter pages with 'Germany'
        pages_with_germany = extract_text_from_pdf(st.session_state.uploaded_file)
        
        if pages_with_germany:
            st.write(f"Found 'Germany' in {len(pages_with_germany)} pages. Processing relevant pages...")
            
            st.session_state.uploaded_file.seek(0)  # Reset file pointer
            images = convert_from_bytes(st.session_state.uploaded_file.read())
            
            results = {}
            
            with st.expander("Relevant Slides:"):
                for page_num in pages_with_germany.keys():
                    image = images[page_num]
                    st.image(image, caption=f"Page {page_num+1}", use_container_width=True)
                    
                    # Step 3: Process images with Llama Vision via Together API
                    extracted_content = process_image_with_llama(image)
                    results[page_num + 1] = extracted_content if extracted_content else None
            
            with st.expander("Parsed JSON file:"):
                # Step 4: Export as JSON
                extracted_data = {"pages": results}
                json_data = json.dumps(extracted_data, indent=4)
                st.json(extracted_data)

            with st.expander("Output File:"):
                # Step 5: Structure extracted data
                try:
                    structured_data = structure_extracted_data(json_data)
                    df = pd.read_json(io.StringIO(structured_data), orient='columns')
                    st.write("### Structured Financial Data")
                    st.dataframe(df)
                    st.session_state.df = df
                except:
                    st.write("Failed to structure data.")
            
        else:
            st.write("No relevant information found.")

    if st.session_state.df is not None:
        csv_buffer = io.StringIO()
        st.session_state.df.to_csv(csv_buffer, index=False)
        csv_bytes = io.BytesIO(csv_buffer.getvalue().encode())
        
        if st.download_button(
            label="Download Financial Data CSV",
            data=csv_bytes,
            file_name="germany_financial_data.csv",
            mime="text/csv",
            use_container_width=True, type='primary'
        ):
            st.session_state["uploaded_file"] = None


if __name__ == "__main__":
    main()
