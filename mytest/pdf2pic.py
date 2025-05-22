import os
import fitz  # PyMuPDF

def pdf_to_images(pdf_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        out_path = os.path.join(output_dir, f"{base}_page{page_num+1}.png")
        pix.save(out_path)
        print(f"Saved: {out_path}")
    doc.close()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("用法: python pdf2pic.py <pdf文件路径> <输出文件夹>")
    else:
        pdf_to_images(sys.argv[1], sys.argv[2])
