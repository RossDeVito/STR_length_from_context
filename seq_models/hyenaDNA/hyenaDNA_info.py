# First ID of soft prompt tokens. Is the same as the initial
# embedding matrix size (i.e., number of original tokens).
PROMPT_START_ID = 16

# Maximum sequence length for heyenaDNA models
MAX_SEQ_LENGTH = {
	'LongSafari/hyenadna-tiny-1k-seqlen-hf': 1024,
	'LongSafari/hyenadna-tiny-1k-seqlen-d256-hf': 1024,
	'LongSafari/hyenadna-small-32k-seqlen-hf': 32768,
	'LongSafari/hyenadna-medium-160k-seqlen-hf': 160000,
	'LongSafari/hyenadna-medium-450k-seqlen-hf': 450000,
	'LongSafari/hyenadna-large-1m-seqlen-hf': 1_000_000,
}