from vllm import LLM, SamplingParams
import os
import time
import faulthandler
faulthandler.enable()

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

smoe = LLM(model="NickyNicky/Mixtral-4x1.1B-TinyDolphin-2.8-1.1b_oasst2_chatML_Cluster",
          max_num_seqs=2, gpu_memory_utilization=0.25)

start_time = time.time()
for _ in range(10):
    outputs = smoe.generate(prompts, sampling_params)
end_time = time.time()

print(f"Generated {len(outputs)} completions in {(end_time - start_time) / 10:.2f} seconds")

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

while True:
    time.sleep(1000)
