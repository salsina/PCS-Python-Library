def trim_text(response):
    return response.strip("\n ")

def extract_label(response, labels):
    response = response.lower()

    if "<label>" in response:
        label_section = response.split("<label>")[1]
    elif "label:" in response:
        label_section = response.split("label:")[1]
    else:
        return None
    
    for label in labels:
        if label.lower() in label_section:
            return label
            
    return None


def extract_reasoning(response):
    reasons = []
    response = response.lower()
    if "<reasoning>" in response and "</reasoning>" in response:
        reasoning_section = response.split("<reasoning>")[1].split("</reasoning>")[0]
        reasons = reasoning_section.strip().split('\n')
    
    elif "reasoning:" in response:
        reasoning_section = response.split("reasoning:")[1]
        reasons = reasoning_section.strip().split('\n')
    
    return reasons

def extract_judgement(response):
    lines = response.split('\n')
    for line in lines:
        if line.strip():
            if "no" in line.lower():
                return "No"
            elif "yes" in line.lower(): 
                return "Yes"
    
    return "Undefined"