def split_syllogism(text: str | list) -> list[str]:
    """
    Split a syllogism into individual sentences (propositions).
    Handles string input (splitting by ". " and normalizing Chinese punctuation)
    and list input (returning as is).
    """
    if isinstance(text, list):
        # Already split
        return [s for s in text if isinstance(s, str) and s.strip()]

    if not isinstance(text, str):
        text = str(text)

    # Normalize Chinese punctuation to English style
    text_normalized = text.replace("。", ". ")

    # Normalize Bengali punctuation to English style
    text_normalized = text_normalized.replace("।", ". ")

    # Remove trailing dot if present to avoid empty string at end
    stripped_text = text_normalized.strip()
    if stripped_text.endswith("."):
        stripped_text = stripped_text[:-1]

    # Split by ". "
    parts = stripped_text.split(". ")

    # Clean empty sentences
    sentences = [s for s in parts if s.strip()]

    return sentences
