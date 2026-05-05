#Packages Import
import json
import os
import re

#Transcripts local Folders
raw_root = "../Data/Transcripts/Raw"
out_root = "../Data/Transcripts/Normalized"

#Populating metadata for the selected companies

COMPANY_METADATA = {
    "APA": {
        "company_name": "APA Corporation",
        "sector": "Energy",
        "market_tier": "Low",
        "exchange": "NASDAQ",
        "country": "USA"
    },
     "XOM": {
        "company_name": "Exxon Mobil Corporation",
        "sector": "Energy",
        "market_tier": "High",
        "exchange": "NYSE",
        "country": "USA"
    },
     "BKR": {
        "company_name": "Baker Hughes Company",
        "sector": "Energy",
        "market_tier": "Medium",
        "exchange": "NASDAQ",
        "country": "USA"
    },
     "INTC": {
        "company_name": "Intel",
        "sector": "Information Technology",
        "market_tier": "High",
        "exchange": "NASDAQ",
        "country": "USA"
     },
     "ADSK": {
        "company_name": "Autodesk Inc.",
        "sector": "Information Technology",
        "market_tier": "Medium",
        "exchange": "NASDAQ",
        "country": "USA"
    },
     "HPE": {
        "company_name": "Hewlett Packard Enterprise",
        "sector": "Information Technology",
        "market_tier": "Low",
        "exchange": "NYSE",
        "country": "USA"
    },
     "GS": {
        "company_name": "Goldman Sachs",
        "sector": "Financials",
        "market_tier": "High",
        "exchange": "NYSE",
        "country": "USA"
     },
     "BX": {
        "company_name": "Blackstone Inc",
        "sector": "Financials",
        "market_tier": "Medium",
        "exchange": "NYSE",
        "country": "USA"
    },
     "PYPL": {
        "company_name": "PayPal Holdings Inc",
        "sector": "Financials",
        "market_tier": "Medium",
        "exchange": "NASDAQ",
        "country": "USA"
    },
     "WMT": {
        "company_name": "Walmart",
        "sector": "Consumer Discretionary",
        "market_tier": "High",
        "exchange": "NYSE",
        "country": "USA"
     },
     "DAL": {
        "company_name": "Delta Air Lines Inc.",
        "sector": "Consumer Discretionary",
        "market_tier": "Medium",
        "exchange": "NYSE",
        "country": "USA"
    },
     "APTV": {
        "company_name": "Aptiv PLC",
        "sector": "Consumer Discretionary",
        "market_tier": "Low",
        "exchange": "NYSE",
        "country": "USA"
    },
     "GOOGL": {
        "company_name": "Alphabet Inc.",
        "sector": "Communication Services",
        "market_tier": "High",
        "exchange": "NASDAQ",
        "country": "USA"
    },
     "CMCSA": {
        "company_name": "Comcast",
        "sector": "Communication Services",
        "market_tier": "Medium",
        "exchange": "NASDAQ",
        "country": "USA"
     },
     "FOX": {
        "company_name": "Fox Corporation",
        "sector": "Communication Services",
        "market_tier": "Low",
        "exchange": "NASDAQ",
        "country": "USA"
    }
}

#Source Metadata
SOURCE_NAME = "University Database"

#Function to acquire the required schema variable items.
def normalize_single_transcript(raw, ticker, year, quarter):
    # Retrieving metadata from existing mapping
    meta = COMPANY_METADATA.get(ticker, {})
    raw_split = raw.get("transcript_split", [])

    # Q&A Boundary Detection
    qa_start_index = len(raw_split)
    found_trigger = False
    for i, item in enumerate(raw_split):
        text_lower = item.get("text", "").lower()
        if any(ot in text_lower for ot in ["open up", "take questions", "first question", "begin the question"]):
            found_trigger = True
        # Strictly anchor to the first Operator/Moderator turn after the trigger
        if found_trigger and (
                "operator" in item.get("speaker", "").lower() or "moderator" in item.get("speaker", "").lower()):
            qa_start_index = i
            break

    # Executive Whitelist - captures everyone who speaks in the Prepared Remarks.
    management_team = set()
    management_last_names = set()
    for i in range(qa_start_index):
        spk = raw_split[i].get("speaker", "Unknown")
        if "operator" not in spk.lower() and spk != "Unknown":
            management_team.add(spk)
            if len(spk.split()) > 0:
                management_team.add(spk.split()[0])  # First name (Jon)
                management_last_names.add(spk.split()[-1])  # Last name (Gray)

    #Hierarchy Parser
    sections_prepared, sections_qa = [], []
    last_introduced_analyst = None

    for i, item in enumerate(raw_split):
        speaker = item.get("speaker", "Unknown").strip()
        text = item.get("text", "").strip()

        is_qa_phase = i >= qa_start_index
        speaker_lower = speaker.lower()
        text_lower = text.lower()

        # Moderator Identity Extraction
        if "operator" in speaker_lower or "moderator" in speaker_lower:
            role = "moderator"
            # Regex captures analyst names in standard intro formats
            match = re.search(r"(?:to|from|with|of)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", text)
            if match:
                last_introduced_analyst = match.group(1).strip()
        # Analyst Lockdown
        elif is_qa_phase and last_introduced_analyst and last_introduced_analyst in speaker:
            role = "analyst"

        # Executives
        elif any(boss in speaker for boss in management_team) or (
                len(speaker.split()) > 0 and speaker.split()[-1] in management_last_names):
            role = "executive"

        elif is_qa_phase:
            is_internal = any(
                m in text_lower[:100] for m in ["our strategy", "our results", "our business", "my colleagues"])
            role = "executive" if is_internal else "analyst"
        else:
            role = "executive"

        # Resolves malformed speaker fields
        is_malformed = (
                    len(speaker.split()) > 6 or "$" in speaker or "docsis" in speaker_lower or "4.0" in speaker_lower)
        if is_malformed and i > 0:
            text = f"{speaker} {text}"
            prev_sec = sections_qa if is_qa_phase else sections_prepared
            # Attribute text back to previous person
            speaker = prev_sec[-1]["speaker"] if prev_sec else "Unknown"

        # Cleanup Operator metadata leaked into text
        if "Operator:" in text:
            text = text.split("Operator:")[0].strip()

        entry = {"speaker": speaker, "role": role, "text": text}
        if is_qa_phase:
            sections_qa.append(entry)
        else:
            sections_prepared.append(entry)

    # Full Text Aggregation
    full_text_list = [c['text'] for c in sections_prepared] + [c['text'] for c in sections_qa]
    full_transcript_text = " ".join(full_text_list)

    # Schema Assembly


    return {
    "metadata": {
        "ticker": ticker,
        "company_name": meta.get("company_name", "Unknown"),
        "sector": meta.get("sector", "Unknown"),
        "fiscal_year": int(year),
        "fiscal_quarter": quarter,
        "calendar_date": raw.get("date"),
        "source": SOURCE_NAME
    },
    "transcript": {
        "sections": [
            {"section_type": "prepared_remarks", "order": 1, "content": sections_prepared},
            {"section_type": "qa", "order": 2, "content": sections_qa}
        ],
        "full_transcript_text": full_transcript_text
    }
}

#Metadata for the raw transcripts includes : Date, Transcript, Transcript Split (Speaker Segmentation )
#Converting the raw transcripts to fit into the above schema

def run_normalization():
    for ticker in os.listdir(raw_root):
        ticker_path = os.path.join(raw_root, ticker)
        if not os.path.isdir(ticker_path):
            continue

        for year in os.listdir(ticker_path):
            year_path = os.path.join(ticker_path, year)
            if not os.path.isdir(year_path):
                continue

            for file in os.listdir(year_path):
                if not file.endswith(".json"):
                    continue

                match = re.search(r"_(\d)\.json$", file)
                if not match:
                    print(f"Skipping file with unexpected name: {file}")
                    continue

                quarter = f"Q{match.group(1)}"

                with open(os.path.join(year_path, file), "r", encoding="utf-8") as f:
                    raw = json.load(f)

                schema = normalize_single_transcript(raw, ticker, year, quarter)

                out_dir = os.path.join(out_root, ticker, year)
                os.makedirs(out_dir, exist_ok=True)

                out_path = os.path.join(out_dir, f"{quarter}.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(schema, f, indent=2)



if __name__ == "__main__":
    run_normalization()