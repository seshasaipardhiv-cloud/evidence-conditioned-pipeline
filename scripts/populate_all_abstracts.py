import json
import urllib.request
import xml.etree.ElementTree as ET
import time
import re

with open('evidence/processed/papers.jsonl', 'r', encoding='utf-8') as f:
    papers = [json.loads(line) for line in f if line.strip()]

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

# Handle specific DOIs
for p in papers:
    if p.get('paper_id') == 'paper_10.1038_s42256-023-00633-5':
        p['pmid'] = '37693852'
    elif p.get('paper_id') == 'paper_10.3390_bioengineering11010013':
        p['title'] = 'Multi-Modal Ensemble Deep Learning in Head and Neck Cancer HPV Sub-Typing'

updated = 0
for i, p in enumerate(papers):
    pmid = p.get('pmid')
    curr_ab = p.get('abstract')
    if not curr_ab and pmid:
        url = f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pmid}&retmode=xml'
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                root = ET.fromstring(resp.read().decode('utf-8'))
                t_elem = root.find('.//ArticleTitle')
                if t_elem is not None and t_elem.text:
                    p['title'] = t_elem.text.strip()
                abs_texts = [elem.text for elem in root.findall('.//AbstractText') if elem.text]
                if abs_texts:
                    full_ab = ' '.join(abs_texts).strip()
                    p['abstract'] = full_ab
                    p['abstract_available'] = True
                    updated += 1
                    print(f"[{i+1}/30] Saved abstract for PMID {pmid} ({len(full_ab)} chars)")
        except Exception as e:
            print(f"Error fetching PMID {pmid}: {e}")
        time.sleep(0.3)
    elif not curr_ab and p.get('doi') == '10.3390/bioengineering11010013':
        url = 'https://api.crossref.org/works/10.3390/bioengineering11010013'
        req = urllib.request.Request(url, headers={'User-Agent': 'ResearchAudit/1.0 (mailto:audit@example.com)'})
        try:
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode('utf-8'))['message']
                raw_ab = data.get('abstract', '')
                clean_ab = re.sub('<[^<]+?>', '', raw_ab).strip()
                p['abstract'] = clean_ab
                p['abstract_available'] = True
                updated += 1
                print(f"[{i+1}/30] Saved CrossRef abstract for {p.get('doi')} ({len(clean_ab)} chars)")
        except Exception as e:
            print(f"CrossRef error: {e}")

with open('evidence/processed/papers.jsonl', 'w', encoding='utf-8') as f:
    for p in papers:
        f.write(json.dumps(p) + '\n')

print(f"Done! Successfully updated {updated} papers. Total papers: {len(papers)}")
with_abs = sum(1 for p in papers if p.get('abstract'))
print(f"Papers with genuine abstract text: {with_abs} / {len(papers)}")
