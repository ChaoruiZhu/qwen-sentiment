import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    
    model_name = "Qwen/Qwen3-0.6B"  
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
    model.to(device)
    model.eval()

    
    dataset = load_dataset("imdb", split="test", streaming=True).take(5)

    print("\n--- Starting Inference (Logits Forced Choice) ---")

    
    pos_ids = tokenizer(" positive", add_special_tokens=False)["input_ids"]
    neg_ids = tokenizer(" negative", add_special_tokens=False)["input_ids"]

   
    for i, entry in enumerate(dataset):
        review_text = entry["text"][:500]
        actual = "positive" if entry["label"] == 1 else "negative"

        prompt = (
            "You are a sentiment classifier.\n"
            "Answer with ONLY one word: positive or negative.\n\n"
            f"Review:\n{review_text}\n\n"
            "Sentiment:"
        )

        
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.no_grad():
            
            def score_candidate(candidate_token_ids):
                """
                给定候选答案的 token ids，返回把它接在 prompt 后面时的 log-prob 总分
                """
                cand = torch.tensor([candidate_token_ids], device=device)
                full_ids = torch.cat([input_ids, cand], dim=1)

                if attention_mask is not None:
                    cand_mask = torch.ones_like(cand, device=device)
                    full_mask = torch.cat([attention_mask, cand_mask], dim=1)
                else:
                    full_mask = None

                out = model(input_ids=full_ids, attention_mask=full_mask)
                logits = out.logits  # [1, seq_len, vocab]

                
                prompt_len = input_ids.shape[1]
                total_logprob = 0.0

                for j, tok_id in enumerate(candidate_token_ids):
                    
                    pos = prompt_len + j - 1
                    step_logits = logits[0, pos, :]  # [vocab]
                    log_probs = torch.log_softmax(step_logits, dim=-1)
                    total_logprob += log_probs[tok_id].item()

                return total_logprob

            pos_score = score_candidate(pos_ids)
            neg_score = score_candidate(neg_ids)

        pred = "positive" if pos_score > neg_score else "negative"

        print(f"Review {i+1}:")
        print(f"Actual:    {actual}")
        print(f"Predicted: {pred}")
        print(f"Scores:    pos={pos_score:.4f}, neg={neg_score:.4f}")
        print("-" * 40)

if __name__ == "__main__":
    main()

