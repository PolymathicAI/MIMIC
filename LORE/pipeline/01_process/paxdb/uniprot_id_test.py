#!/usr/bin/env python3
"""
Simple test script to check UniProt API response format. This is a one-off script that was used during development of 01_find_unmapped_names.py, and does not need to be run regularly.
"""

import requests
import json

def test_uniprot_api():
    """Test the UniProt API with a simple example"""
    
    # Test data
    gene_names = ["TP53", "BRCA1"]
    taxonomy_id = "9606"  # Human
    
    base_url = "https://rest.uniprot.org/idmapping"
    
    # Submit job
    data = {
        "from": "Gene_Name",
        "to": "UniProtKB",
        "ids": " ".join(gene_names),
        "taxId": taxonomy_id
    }
    
    print("Submitting job...")
    print(f"Data: {data}")
    
    try:
        response = requests.post(f"{base_url}/run", data=data)
        print(f"Submit response status: {response.status_code}")
        print(f"Submit response headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            response_data = response.json()
            print(f"Submit response JSON: {json.dumps(response_data, indent=2)}")
            
            # Try to get job ID
            job_id = None
            for possible_key in ["jobId", "id", "job_id"]:
                if possible_key in response_data:
                    job_id = response_data[possible_key]
                    break
            
            if job_id:
                print(f"Job ID: {job_id}")
                
                # Check status
                print("\nChecking job status...")
                status_response = requests.get(f"{base_url}/status/{job_id}")
                print(f"Status response status: {status_response.status_code}")
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    print(f"Status response JSON: {json.dumps(status_data, indent=2)}")
                else:
                    print(f"Status response text: {status_response.text}")
            else:
                print("Could not find job ID in response")
        else:
            print(f"Submit response text: {response.text}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_uniprot_api()