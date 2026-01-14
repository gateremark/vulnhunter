"""
VulnHunter Model Testing Script
Test the trained security agent on real vulnerable code examples.
"""
from unsloth import FastLanguageModel
import torch

# Load the model from HuggingFace
print("Loading VulnHunter model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    "gateremark/vulnhunter-agent",
    max_seq_length=2048,
    load_in_4bit=True,  # Use 4-bit for faster inference
)
FastLanguageModel.for_inference(model)
print("Model loaded!")


def analyze_code(code: str) -> str:
    """Analyze code for security vulnerabilities."""
    prompt = f"""You are VulnHunter, an AI security researcher. Your task is to find and patch security vulnerabilities in web applications.

When analyzing code, look for:
- SQL Injection: Unsanitized input in SQL queries
- XSS: Unescaped user input in HTML
- Path Traversal: Unchecked file paths

Respond with your analysis and a JSON action like:
{{"identify_vuln": {{"type": "sql_injection", "file": "app.py", "line": 45}}}}

Analyze this code:
```python
{code}
```

What vulnerability exists and how would you fix it?"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response


# Test cases
print("\n" + "="*60)
print("TEST 1: SQL Injection")
print("="*60)

sql_vuln_code = '''
@app.route("/login", methods=["POST"])
def login():
    username = request.form.get("username")
    password = request.form.get("password")
    query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'"
    cursor.execute(query)
    return "OK" if cursor.fetchone() else "Failed"
'''

print("Code:")
print(sql_vuln_code)
print("\nVulnHunter Analysis:")
print(analyze_code(sql_vuln_code))


print("\n" + "="*60)
print("TEST 2: XSS Vulnerability")
print("="*60)

xss_vuln_code = '''
@app.route("/search")
def search():
    query = request.args.get("q", "")
    return f"<h1>Results for: {query}</h1>"
'''

print("Code:")
print(xss_vuln_code)
print("\nVulnHunter Analysis:")
print(analyze_code(xss_vuln_code))


print("\n" + "="*60)
print("TEST 3: Path Traversal")
print("="*60)

path_vuln_code = '''
@app.route("/download/<filename>")
def download(filename):
    return send_file(f"/uploads/{filename}")
'''

print("Code:")
print(path_vuln_code)
print("\nVulnHunter Analysis:")
print(analyze_code(path_vuln_code))


print("\n" + "="*60)
print("Testing Complete!")
print("="*60)
