# gpu_manager.py
import ray
import time
from moe_train import run_moe_rlhf_task

# 1. Ray 초기화 (현재 머신의 모든 GPU를 감지하여 클러스터 형성)
# 대규모 클러스터라면 ray.init(address='auto')로 연결
ray.init(ignore_reinit_error=True)

print(f"Total GPUs detected: {ray.available_resources().get('GPU', 0)}")

# 2. Worker 정의 (GPU를 요구하는 Actor)
@ray.remote(num_gpus=1) # 중요: 이 작업은 GPU 1개를 전용으로 씀을 선언
def worker_process(config):
    try:
        result = run_moe_rlhf_task(config)
        return result
    except Exception as e:
        return {"job_id": config['job_id'], "status": "failed", "error": str(e)}

# 3. 작업 대기열 (Job Queue) 생성
# 예: 다양한 Learning Rate로 실험을 10개 돌리고 싶음
experiments_queue = [
    {"job_id": i, "lr": 1e-5 * (i+1), "epochs": 3} 
    for i in range(10) 
]

# 4. 스케줄링 로직 (빈 GPU 자동 할당)
# Ray는 .remote()를 호출하면 즉시 실행하지 않고, 
# 'num_gpus' 요구사항이 충족될 때까지 자동으로 Pending 상태로 대기시킵니다.

print("--- 스케줄링 시작: GPU가 비는대로 작업이 투입됩니다 ---")

# 모든 작업을 일단 Ray 스케줄러에 던집니다 (비동기 제출)
pending_futures = []
for job_config in experiments_queue:
    # 4개의 GPU가 있고 10개의 작업을 던지면, 
    # 4개는 즉시 실행(Running), 6개는 대기(Pending) 상태가 됨.
    # 하나가 끝나면 즉시 다음 것이 실행됨.
    future = worker_process.remote(job_config)
    pending_futures.append(future)

# 5. 결과 모니터링 (작업이 끝나는 순서대로 결과 출력)
while pending_futures:
    # 완료된 작업(ready)과 아직 도는 작업(not_ready)을 분리
    ready_ids, pending_futures = ray.wait(pending_futures)
    
    # 완료된 작업의 결과 가져오기
    results = ray.get(ready_ids)
    
    for res in results:
        print(f"🎉 Job Finished: {res}")
        # 여기서 MLflow나 WandB로 로그 전송 가능

print("모든 MoE 학습 스케줄 종료.")
ray.shutdown()
