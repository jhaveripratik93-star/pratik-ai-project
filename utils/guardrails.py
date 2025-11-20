import re
from typing import List, Tuple

def validate_user_input(user_input: str) -> Tuple[bool, str]:
    """
    Validates user input for security and content appropriateness.
    
    Args:
        user_input (str): The user's input text
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    
    # Check if input is empty or too short
    if not user_input or len(user_input.strip()) < 2:
        return False, "Please enter a valid question or message."
    
    # Check for excessive length
    if len(user_input) > 1000:
        return False, "Message too long. Please keep it under 1000 characters."
    
    # Check for malicious patterns
    malicious_patterns = [
        r'<script.*?>.*?</script>',  # Script tags
        r'javascript:',              # JavaScript protocol
        r'eval\s*\(',               # eval function
        r'exec\s*\(',               # exec function
        r'import\s+os',             # OS imports
        r'__import__',              # Dynamic imports
        r'subprocess',              # Subprocess calls
    ]
    
    for pattern in malicious_patterns:
        if re.search(pattern, user_input, re.IGNORECASE):
            return False, "Invalid input detected. Please ask a legitimate question."
    
    # Check for inappropriate content
    inappropriate_words = [
        'hack', 'exploit', 'malware', 'virus', 'attack', 'breach', 
        'password', 'credential', 'token', 'secret', 'private key'
    ]
    
    for word in inappropriate_words:
        if word in user_input.lower():
            return False, "Please ask questions related to network troubleshooting only."
    
    return True, ""

def sanitize_input(user_input: str) -> str:
    """
    Sanitizes user input by removing potentially harmful characters.
    
    Args:
        user_input (str): Raw user input
        
    Returns:
        str: Sanitized input
    """
    # Remove HTML tags
    user_input = re.sub(r'<[^>]+>', '', user_input)
    
    # Remove special characters that could be used for injection
    user_input = re.sub(r'[<>"\';\\]', '', user_input)
    
    # Normalize whitespace
    user_input = ' '.join(user_input.split())
    
    return user_input.strip()

def is_network_related(user_input: str) -> bool:
    """
    Checks if the user input is related to network troubleshooting.
    
    Args:
        user_input (str): User's input text
        
    Returns:
        bool: True if network-related, False otherwise
    """
    network_keywords = [
        'network', 'connection', 'internet', 'wifi', 'ethernet', 'router',
        'switch', 'firewall', 'dns', 'ip', 'tcp', 'udp', 'ping', 'traceroute',
        'bandwidth', 'latency', 'packet', 'protocol', 'port', 'gateway',
        'subnet', 'vlan', 'vpn', 'ssl', 'tls', 'http', 'https', 'ftp',
        'troubleshoot', 'connectivity', 'outage', 'slow', 'timeout', 'server',
        'client', 'host', 'domain', 'url', 'web', 'site', 'online', 'offline',
        'connect', 'disconnect', 'access', 'error', 'issue', 'problem',
        'fix', 'solve', 'help', 'how', 'what', 'why', 'when', 'where'
    ]
    
    user_input_lower = user_input.lower()
    # More lenient check - if it's a short question or contains common question words, allow it
    if len(user_input.split()) <= 5 or any(word in user_input_lower for word in ['how', 'what', 'why', 'help', 'can', 'do']):
        return True
    
    return any(keyword in user_input_lower for keyword in network_keywords)