import torch
import transformers as tr
from torch.nn import functional as F
import concurrent.futures
import math

amateur_path = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
expert_path = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

tokenizer = tr.AutoTokenizer.from_pretrained(amateur_path)
amateur_model = tr.AutoModelForCausalLM.from_pretrained(amateur_path)
expert_model = tr.AutoModelForCausalLM.from_pretrained(expert_path)


user_message = """Give a very very brief docstring for the following function:\n```\nfunction updateEloScores(
	scores,
	results,
	kFactor = 4,
) {
	for (const result of results) {
		const { first, second, outcome } = result;
		const firstScore = scores[first] ?? 1000;
		const secondScore = scores[second] ?? 1000;

		const expectedScoreFirst = 1 / (1 + Math.pow(10, (secondScore - firstScore) / 400));
		const expectedScoreSecond = 1 / (1 + Math.pow(10, (firstScore - secondScore) / 400));
		let sa = 0.5;
		if (outcome === 1) {
			sa = 1;
		} else if (outcome === -1) {
			sa = 0;
		}
		scores[first] = firstScore + kFactor * (sa - expectedScoreFirst);
		scores[second] = secondScore + kFactor * (1 - sa - expectedScoreSecond);
	}
	return scores;
}\n```"""

prompt = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": user_message},
    ],
    add_generation_prompt=True,
    tokenize=False,
)

def prime_kv_cache(model, input_ids):
    with torch.inference_mode():
        out = model(input_ids, use_cache=True)
    return out.past_key_values, out.logits[:, -1, :]


def get_logits(model, input_ids, past_key_values):
	with torch.inference_mode():
		outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
		return outputs.logits[:, -1, :], outputs.past_key_values


def contrastive_generation(
    amateur, 
    expert, 
    tokenizer, 
    prompt, 
    max_tokens, 
    alpha=0.1,              
    top_k=None,             
    tau=1.0,                
    beta=1.0,               
    amateur_last_token_only=True,  
    stop_on_eos=True,
    device=None):

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    expert.eval()
	amateur.eval()
    expert.to(device)
	amateur.to(device)

    initial_input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated_ids = initial_input_ids.clone()

	# This builds kv cache for further optimization
    expert_past, _ = prime_kv_cache(expert, initial_input_ids)
    if amateur_last_token_only:
        amateur_past = None  
    else:
        amateur_past, _ = prime_kv_cache(amateur, initial_input_ids)

	# Feed the last token when we step through with a cache
    for _ in range(max_tokens):
        current_token = generated_ids[:, -1:]

        expert_logits, expert_past = get_logits(expert, current_token, expert_past)
        log_pe = F.log_softmax(expert_logits, dim=-1)  

        amateur_past_step = None if amateur_last_token_only else amateur_past
        amateur_logits, amateur_past = get_logits(amateur, current_token, amateur_past_step)
        log_pa = F.log_softmax(amateur_logits / tau, dim=-1)

        # keep tokens where log_pe >= log_max + log(alpha), according to the paper
        log_max = log_pe.max(dim=-1, keepdim=True).values  # (1,1)
        thresh = log_max + math.log(alpha)
        keep_mask = log_pe >= thresh  # (1, V) 

        if top_k is not None:
            _, idx = torch.topk(log_pe, k=min(top_k, log_pe.size(-1)), dim=-1)
            cap_mask = torch.zeros_like(keep_mask, dtype=torch.bool)
            cap_mask.scatter_(1, idx, True)
            keep_mask = keep_mask & cap_mask

        # fall back to top-1 expert if alpha too low
        if not torch.any(keep_mask):
            keep_mask = log_pe == log_max

        # mask out implausible tokens after contrastive scoring
        scores = log_pe - beta * log_pa
        scores[~keep_mask] = -float("inf")

        probs = F.softmax(scores, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # (1,1)

        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        if stop_on_eos and tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


if __name__ == "__main__":
    generated_text = contrastive_generation(
        amateur=amateur_model,
        expert=expert_model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=60,
        alpha=0.5,
        top_k=20,
        tau=0.95
    )
    print(generated_text)

