import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    model_path = r"C:\Users\wzh\Desktop\deepseek-r1-distill-qwen-1.5b"
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    messages = [
        {"role": "system", "content": "你是一个严谨的中文问答助手。只输出最终答案，不要展开推理。"},
        {
            "role": "user",
            "content": "根据规则：P0表示核心服务完全不可用或数据丢失风险，要求15分钟内响应、2小时内给出临时恢复方案。问题：P0服务故障多久内响应？",
        },
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=96,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    answer = tokenizer.decode(output[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
    print(json.dumps({"answer": answer}, ensure_ascii=False))


if __name__ == "__main__":
    main()
