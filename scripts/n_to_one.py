import fitz, os # PyMuPDF
from glob import glob

def un_n_up_pdf(input_pdf_path, output_pdf_path):
    doc = fitz.open(input_pdf_path)
    new_doc = fitz.open() # Create a new PDF

    num_original_sheets = len(doc) # Total number of physical sheets in the input PDF

    for page_num, page in enumerate(doc): # 'page' here refers to a sheet from the input PDF
        # Calculate the mid-point for splitting, as a reference
        mid_x = page.rect.width / 2

        # Check if this is the last sheet AND the total number of sheets is odd
        is_last_sheet = (page_num == num_original_sheets - 1)
        is_odd_sheets_total = (num_original_sheets % 2 != 0)

        if is_last_sheet and is_odd_sheets_total:
            print(f"Processing last sheet ({page_num+1}/{num_original_sheets}) by detecting visible content area.")
			# Get the text bounding boxes (this detects where actual content is)
            blocks = page.get_text("blocks")
            if not blocks:
                print("Warning: No content detected on last page.")
                continue

			# Calculate bounding box covering all content
            content_bbox = fitz.Rect(blocks[0][:4])
            for block in blocks[1:]:
                content_bbox |= fitz.Rect(block[:4])  # union of rectangles

			# Expand the content box slightly for padding (optional)
            padding = 10  # points
            content_bbox.x0 = max(0, content_bbox.x0 - padding)
            content_bbox.y0 = max(0, content_bbox.y0 - padding)
            content_bbox.x1 = min(page.rect.width, content_bbox.x1 + padding)
            content_bbox.y1 = min(page.rect.height, content_bbox.y1 + padding)

			# Create a new page with same size as content bounding box
            new_page = new_doc.new_page(width=content_bbox.width, height=content_bbox.height)
            new_page.show_pdf_page(new_page.rect, doc, page_num, clip=content_bbox)
            
        else:
            # Process as a regular 2-up sheet (left and right halves)
            print(f"Processing sheet ({page_num+1}/{num_original_sheets}) as two pages.")
            
            # Define the left half
            left_rect = fitz.Rect(0, 0, mid_x, page.rect.height)
            # Define the right half
            right_rect = fitz.Rect(mid_x, 0, page.rect.width, page.rect.height)

            # Add left page
            new_page_left = new_doc.new_page(width=left_rect.width, height=left_rect.height)
            new_page_left.show_pdf_page(new_page_left.rect, doc, page_num, clip=left_rect)

            # Add right page
            new_page_right = new_doc.new_page(width=right_rect.width, height=right_rect.height)
            new_page_right.show_pdf_page(new_page_right.rect, doc, page_num, clip=right_rect)

    new_doc.save(output_pdf_path)
    new_doc.close()
    doc.close()
    print(f"Successfully converted '{input_pdf_path}' to 1-up format at '{output_pdf_path}'")


file_name = "서울특별시_2024년 지도정보 플랫폼 및 전문활용 연계 시스템 고도화 용.pdf"
file_dir = "pdf_files/" + file_name
hwp_file_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_dir)
pdf_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clean_pdf")
os.makedirs(pdf_output_dir, exist_ok = True)
pdf_output = os.path.join(pdf_output_dir, "clean" + file_name)

un_n_up_pdf(hwp_file_dir, pdf_output)