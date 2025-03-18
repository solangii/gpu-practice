## Goal of profile

1. integrate a CUDA kernel inside a pytorch program
2. Learn how to profile it

---
### some of interesting

1. `pytorch_squre.py`
  - CUDA는 비동기적 특성을 가짐. 
  - 이는 프로파일링 할 때 GPU 작업의 시간을 정확히 측정하는데 문제가 됨. 그래서 `torch.cuda.synchronize()`라는 명령어를 사용. 
  - 이 명령어는 GPU가 모든 일을 끝낼 때까지 기다리게 함. 이렇게 하면 GPU가 일을 다 끝낸 후에 시간을 재기 때문에, 정확한 시간을 측정할 수 있습니다.

2. PyTorch Profiler
  - Memcpy HtoD (Pageable -> device)
  - **H**ost **to** **D**evice copy
  - Pageable memory is on host but can be copied freely in out of RAM

3. Custom C++ Extetion w/ Pytorch
  - use `load_inline()`. build ninza참고. 알아서 컴파일 다 해줌.