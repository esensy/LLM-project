import os
import win32com.client
import pythoncom 
from glob import glob

def convert_hwp_to_pdf_robust(hwp_path, pdf_path):
    """
    Opens an HWP file using Hancom HWP application and saves it as a PDF.
    Includes robustness and attempts to fix SaveAs parameters.

    Args:
        hwp_path (str): The full path to the input HWP file.
        pdf_path (str): The full path for the output PDF file.
    """
    HwpCtrl = None
    try:
        pythoncom.CoInitialize()
        HwpCtrl = win32com.client.gencache.EnsureDispatch("HWPFrame.HwpObject")
        HwpCtrl.XHwpWindows.Item(0).Visible = True

        print(f"Attempting to open: {os.path.basename(hwp_path)}")

        HwpCtrl.Open(hwp_path)

        print(f"Successfully opened: {os.path.basename(hwp_path)}")
        print(f"Attempting to save as PDF: {os.path.basename(pdf_path)}")

        HwpCtrl.SaveAs(pdf_path, "PDF", "download:true")

        print(f"Successfully saved as PDF: {os.path.basename(pdf_path)}")
        print("Attempting to clear/close document...")

        HwpCtrl.Clear()

        print("Document cleared.")

        print("Attempting to quit HWP application...")
        HwpCtrl.Quit()
        print("HWP application quit successfully.")

    except Exception as e:
        print(f"\nAn error occurred while processing {os.path.basename(hwp_path)}:")
        print(e)
        if HwpCtrl:
             try:
                 print("Attempting cleanup quit in error handler...")
                 HwpCtrl.Quit()
                 print("Cleanup quit successful.")
             except Exception as quit_e:
                 print(f"Error during HWP cleanup quit (Quit call in error handler): {quit_e}")

    finally:
        pythoncom.CoUninitialize()

hwp_files_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "files") # HWP 파일 있는 경로.
pdf_output_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdf_files") # PDF 파일 저장할 경로.

# 경로.
os.makedirs(pdf_output_directory, exist_ok = True)

# Get a list of all HWP files in the directory
# Use glob for potentially better handling of varied characters/encodings in filenames
hwp_files = glob(os.path.join(hwp_files_dir, "*.hwp"))

# 모든 HWP 파일 PDF으로 변환하기.
for hwp_full_path in hwp_files:
	# 파일명이 "가나다.hwp"니까 "가나다"만 뽑기.
    base_name = os.path.splitext(os.path.basename(hwp_full_path))[0] 
    pdf_output_path = os.path.join(pdf_output_directory, f"{base_name}.pdf") # "가나다.pdf".

    convert_hwp_to_pdf_robust(hwp_full_path, pdf_output_path)

print("\nConversion process finished.")