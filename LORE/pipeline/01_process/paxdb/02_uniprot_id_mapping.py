#!/usr/bin/env python3
"""
Script to map species taxonomy IDs and gene names to UniProt IDs
using the UniProt ID mapping API.
"""

import requests
import time
import json
import csv
from typing import List, Dict, Optional
import argparse

class UniProtMapper:
    def __init__(self):
        self.base_url = "https://rest.uniprot.org/idmapping"
        self.session = requests.Session()
        
    def submit_mapping_job(self, gene_names: List[str], taxonomy_id: str) -> Optional[str]:
        """
        Submit a mapping job to UniProt API
        
        Args:
            gene_names: List of gene names to map
            taxonomy_id: NCBI taxonomy ID
            
        Returns:
            Job ID if successful, None otherwise
        """
        # Join gene names with commas
        ids = ",".join(gene_names)
        
        data = {
            "from": "UniProtKB_AC-ID",
            "to": "UniProtKB",
            "ids": ids,
        }
        
        try:
            response = self.session.post(f"{self.base_url}/run", data=data)
            response.raise_for_status()
            
            # Debug: print the actual response
            response_data = response.json()
            print(f"Submit response: {response_data}")
            
            # Try different possible keys for job ID
            if "jobId" in response_data:
                job_id = response_data["jobId"]
            elif "id" in response_data:
                job_id = response_data["id"]
            elif "job_id" in response_data:
                job_id = response_data["job_id"]
            else:
                print(f"Unknown response format when submitting job: {response_data}")
                return None
                
            print(f"Submitted job {job_id} for taxonomy {taxonomy_id}")
            return job_id
        except requests.exceptions.RequestException as e:
            print(f"Error submitting job for taxonomy {taxonomy_id}: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response when submitting job: {e}")
            return None
    
    def check_job_status(self, job_id: str) -> str:
        """
        Check the status of a mapping job
        
        Args:
            job_id: Job ID returned from submit_mapping_job
            
        Returns:
            Status string ('RUNNING', 'FINISHED', 'ERROR')
        """
        try:
            response = self.session.get(f"{self.base_url}/status/{job_id}")
            response.raise_for_status()
            
            response_data = response.json()
            print(f"Status check for job {job_id}: {response_data}")
            
            # Check if this is a status response or results response
            if "jobStatus" in response_data:
                return response_data["jobStatus"]
            elif "status" in response_data:
                return response_data["status"]
            elif "state" in response_data:
                return response_data["state"]
            elif "results" in response_data or "failedIds" in response_data:
                # This means the job is finished and we got results
                print(f"Job {job_id} completed with results")
                return "FINISHED"
            else:
                print(f"Unknown response format for job {job_id}: {response_data}")
                return "ERROR"
                
        except requests.exceptions.RequestException as e:
            print(f"Error checking job status for {job_id}: {e}")
            return "ERROR"
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response for job {job_id}: {e}")
            return "ERROR"
    
    def get_mapping_results(self, job_id: str) -> Optional[Dict]:
        """
        Get the results of a completed mapping job
        
        Args:
            job_id: Job ID of completed job
            
        Returns:
            Mapping results as dictionary, None if error
        """
        try:
            # Try the stream endpoint first
            response = self.session.get(f"{self.base_url}/stream/{job_id}")
            response.raise_for_status()
            
            result_data = response.json()
            print(f"Stream results for job {job_id}: Found {len(result_data.get('results', []))} results")
            return result_data
            
        except requests.exceptions.RequestException as stream_error:
            print(f"Stream endpoint failed for job {job_id}: {stream_error}")
            
            # Fallback: try the status endpoint which might have results
            try:
                response = self.session.get(f"{self.base_url}/status/{job_id}")
                response.raise_for_status()
                
                result_data = response.json()
                if "results" in result_data:
                    print(f"Status results for job {job_id}: Found {len(result_data.get('results', []))} results")
                    return result_data
                else:
                    print(f"No results found in status response for job {job_id}")
                    return None
                    
            except requests.exceptions.RequestException as status_error:
                print(f"Status endpoint also failed for job {job_id}: {status_error}")
                return None
    
    def wait_for_job_completion(self, job_id: str, max_wait_time: int = 300) -> bool:
        """
        Wait for a job to complete
        
        Args:
            job_id: Job ID to wait for
            max_wait_time: Maximum time to wait in seconds
            
        Returns:
            True if job completed successfully, False otherwise
        """
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            status = self.check_job_status(job_id)
            
            if status == "FINISHED":
                return True
            elif status == "ERROR":
                print(f"Job {job_id} failed")
                return False
            elif status == "RUNNING":
                print(f"Job {job_id} still running...")
                time.sleep(5)  # Wait 5 seconds before checking again
            else:
                print(f"Unknown status for job {job_id}: {status}")
                time.sleep(5)
        
        print(f"Job {job_id} timed out after {max_wait_time} seconds")
        return False
    
    def map_genes_to_uniprot(self, gene_taxonomy_pairs: List[tuple], batch_size: int = 100) -> List[Dict]:
        """
        Map gene names to UniProt IDs for multiple species
        
        Args:
            gene_taxonomy_pairs: List of (gene_name, taxonomy_id) tuples
            batch_size: Number of genes to process per batch
            
        Returns:
            List of mapping results
        """
        # Group genes by taxonomy ID
        taxonomy_genes = {}
        for gene_name, taxonomy_id in gene_taxonomy_pairs:
            if taxonomy_id not in taxonomy_genes:
                taxonomy_genes[taxonomy_id] = []
            taxonomy_genes[taxonomy_id].append(gene_name)
        
        all_results = []
        
        for taxonomy_id, gene_names in taxonomy_genes.items():
            print(f"\nProcessing {len(gene_names)} genes for taxonomy {taxonomy_id}")
            
            # Process genes in batches
            for i in range(0, len(gene_names), batch_size):
                batch_genes = gene_names[i:i + batch_size]
                print(f"Processing batch {i//batch_size + 1} ({len(batch_genes)} genes)")
                
                # Submit job
                job_id = self.submit_mapping_job(batch_genes, taxonomy_id)
                if not job_id:
                    continue
                
                # Wait for completion
                if self.wait_for_job_completion(job_id):
                    # Get results
                    results = self.get_mapping_results(job_id)
                    if results:
                        # Add taxonomy ID to each result
                        for result in results.get("results", []):
                            result["taxonomy_id"] = taxonomy_id
                        all_results.extend(results.get("results", []))
                        
                        # Also report failed IDs
                        failed_ids = results.get("failedIds", [])
                        if failed_ids:
                            print(f"Failed to map {len(failed_ids)} genes: {failed_ids[:10]}{'...' if len(failed_ids) > 10 else ''}")
                
                # Be nice to the API
                time.sleep(1)
        
        return all_results

def load_input_file(filename: str) -> List[tuple]:
    """
    Load gene names and taxonomy IDs from a CSV file
    
    Expected format: gene_name, taxonomy_id
    """
    gene_taxonomy_pairs = []
    
    with open(filename, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        
        # Skip header if present
        first_row = next(reader)
        if not first_row[0].isdigit() and not first_row[1].isdigit():
            # Assume it's a header
            pass
        else:
            # First row is data
            gene_taxonomy_pairs.append((first_row[0], first_row[1]))
        
        for row in reader:
            if len(row) >= 2:
                gene_name = row[0].strip()
                taxonomy_id = row[1].strip()
                gene_taxonomy_pairs.append((gene_name, taxonomy_id))
    
    return gene_taxonomy_pairs

def save_results(results: List[Dict], output_file: str):
    """
    Save mapping results to a CSV file
    """
    if not results:
        print("No results to save")
        return
    
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['gene_name', 'taxonomy_id', 'uniprot_id', 'protein_name', 'organism', 'status']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        for result in results:
            # Handle both successful mappings and failed IDs
            if 'from' in result and 'to' in result:
                # Successful mapping
                to_data = result['to']
                writer.writerow({
                    'gene_name': result.get('from', ''),
                    'taxonomy_id': result.get('taxonomy_id', ''),
                    'uniprot_id': to_data.get('primaryAccession', '') if isinstance(to_data, dict) else str(to_data),
                    'protein_name': to_data.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', '') if isinstance(to_data, dict) else '',
                    'organism': to_data.get('organism', {}).get('scientificName', '') if isinstance(to_data, dict) else '',
                    'status': 'mapped'
                })
            else:
                # Failed mapping - this shouldn't happen in normal results processing
                writer.writerow({
                    'gene_name': str(result),
                    'taxonomy_id': '',
                    'uniprot_id': '',
                    'protein_name': '',
                    'organism': '',
                    'status': 'failed'
                })

def main():
    parser = argparse.ArgumentParser(description="Map gene names to UniProt IDs")
    parser.add_argument("input_file", help="CSV file with gene names and taxonomy IDs")
    parser.add_argument("-o", "--output", default="uniprot_mapping_results.csv", 
                       help="Output CSV file (default: uniprot_mapping_results.csv)")
    parser.add_argument("-b", "--batch-size", type=int, default=100,
                       help="Batch size for API requests (default: 100)")
    
    args = parser.parse_args()
    
    # Load input data
    print(f"Loading input from {args.input_file}")
    gene_taxonomy_pairs = load_input_file(args.input_file)
    print(f"Loaded {len(gene_taxonomy_pairs)} gene-taxonomy pairs")
    
    # Create mapper and process
    mapper = UniProtMapper()
    results = mapper.map_genes_to_uniprot(gene_taxonomy_pairs, args.batch_size)
    
    # Save results
    save_results(results, args.output)
    print(f"\nResults saved to {args.output}")
    print(f"Successfully mapped {len(results)} entries")

if __name__ == "__main__":
    main()