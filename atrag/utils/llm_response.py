import json


def parse_json(resp: str) -> dict:
    """
    Parses JSON data from an LLM response string.
    This function handles cases where the JSON is wrapped in markdown code blocks
    like "```json ... ```" or "``` ... ```".

    Args:
        resp: The LLM response string, potentially containing JSON.
    Returns:
        A dictionary parsed from the JSON string.
    """
    json_str = (resp or "").strip()
    if len(resp) > 6:
        if resp.startswith("```json") and resp.endswith("```"):
            json_str = resp[7:][:-3]
        elif resp.startswith("```") and resp.endswith("```"):
            json_str = resp[3:][:-3]
    return json.loads(json_str)
