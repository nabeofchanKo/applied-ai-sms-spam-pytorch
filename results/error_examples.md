## Baseline Model — Error Examples

### False Positives (Ham → Spam)
- Transaction-related words (e.g., "account", "money") in legitimate messages
- Informal language and abbreviations resembling spam patterns

### False Negatives (Spam → Ham)
- Joke-like or philosophical spam messages
- Lack of explicit promotional keywords or URLs

---

## Transformer Model — Error Examples

### False Positives (Ham → Spam)
- Messages containing phone numbers and imperative phrases (e.g., "RING ME")

### False Negatives (Spam → Ham)
- Implicit-intent spam messages resembling casual statements
- Ambiguity inherent to the dataset rather than model failure