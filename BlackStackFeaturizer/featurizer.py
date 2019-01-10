'''
Featurizer utilizing the classifier created by blackstack
'''

from bs4 import BeautifulSoup
import helpers
import heuristics
import os
import re


def get_features(pdf_html_dir):
    for pdf_html_page in os.listdir(pdf_html_dir):
        path = os.path.join(pdf_html_dir, pdf_html_page)
        pages = []
        with open(path, 'r') as rf:
            soup = BeautifulSoup(rf, 'html.parser')
            tstring = soup.title.string
            # Example title: S1470160X05000063.pdf-0000
            page_number_re = re.match(r'.*-(\d{4})')
            page_number = page_number_re.group()
            merged_areas = helpers.merge_areas(soup.find_all('div', 'ocr_carea'))
            pages.append({
                'page_no': page_number,
                'soup': soup,
                'page': tstring,
                'areas': [helpers.area_summary(area) for area in merged_areas],
                'lines': [line for line in soup.find_all('span', 'ocr_line')]
            })
        page_areas = [page['areas'] for page in pages]
        doc_stats = helpers.summarize_document([area for areas in page_areas for area in areas])




