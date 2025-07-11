## 1. 多卡推理问题

- tensor-parallel-size需能被num_attention_heads整除，例如num_attention_heads=40，则tensor-parallel-size可取1、2、4、8，取3则会运行失败
- 对于AWQ量化的模型，tensor-parallel-size的取值还需要满足一个条件，group_size*tensor-parallel-size需能被intermediate_size整除. https://github.com/vllm-project/vllm/issues/2699
- 多卡推理不一定必须NVLink，可以走PCIe通道，速度也不会有太大的影响？
- [Distributed inference with vLLM](https://developers.redhat.com/articles/2025/02/06/distributed-inference-with-vllm#gpu_parallelism_techniques_in_vllm)

**注意：** num_attention_heads、intermediate_size、group_size可以从模型的config.json文件中找到

## 2. 推理显存不足，可以考虑以下配置

- 设置--enforce-eager参数，禁用cuda-graph，降低显存同时性能会有影响。https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/
- 设置max_model_len，并用一个较低的数值
