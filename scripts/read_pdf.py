from pypdf import PdfReader

reader = PdfReader("data/filings/AAPL_10K_2025.pdf")

text=""
for page in reader.pages:
    text += page.extract_text()

print(text[:3000])
print(len(text), "characters")
