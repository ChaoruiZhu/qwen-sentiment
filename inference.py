import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    # 1. 检查 Mac 的 MPS 加速
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. 加载模型和分词器
    model_name = "Qwen/Qwen3-0.6B" # 也可以使用 Qwen/Qwen3-0.6B-Instruct
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map=device
    )

    # 3. 加载数据集 (Large Movie Review Dataset / IMDB)
    # 取测试集的前 5 条进行演示
    dataset = load_dataset("imdb", split="test", streaming=True).take(5)

    print("\n--- Starting Inference ---")
    
    for i, entry in enumerate(dataset):
        review_text = entry['text'][:500]  # 截取前500字防止过长
        label = "positive" if entry['label'] == 1 else "negative"

        # 构造 Prompt：要求模型只输出标签
        prompt = f"Review: {review_text}\nSentiment (positive or negative):"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # 推理
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=5, 
                temperature=0.1, # 低随机性
                do_sample=False
            )
        
        # 解码输出
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prediction = full_output[len(prompt):].strip().lower()

        print(f"Review {i+1}:")
        print(f"Actual: {label}")
        print(f"Predicted: {prediction}")
        print("-" * 20)
       
if __name__ == "__main__":
    main()




