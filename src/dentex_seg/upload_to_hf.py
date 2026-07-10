import os
from huggingface_hub import HfApi

def upload_weights_to_hf():
    print("허깅페이스 리포지토리에 가중치 업로드를 준비합니다...")
    repo_id = "chemahc94/dentex-tooth-segmentation"
    token = os.environ.get("HF_TOKEN")
    
    if not token:
        # .env 파일에서 토큰 읽기 시도 (로컬 호스트 실행 지원)
        env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env")
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                for line in f:
                    if line.strip().startswith("HF_TOKEN="):
                        token = line.strip().split("=", 1)[1].strip()
                        break
    
    if not token:
        print("경고: HF_TOKEN 환경 변수가 설정되지 않았습니다. 업로드를 건너뜁니다.")
        return

    api = HfApi(token=token)
    
    # 리포지토리 생성 (이미 존재하면 예외를 무시함)
    try:
        api.create_repo(repo_id=repo_id, exist_ok=True)
        print(f"리포지토리 확인 완료: {repo_id}")
    except Exception as e:
        print(f"리포지토리 생성 중 오류(혹은 이미 존재): {e}")

    # weights 폴더 안의 모델들을 업로드
    weights_dir = "weights"
    if not os.path.exists(weights_dir):
        print("업로드할 weights 폴더가 존재하지 않습니다.")
        return

    print("가중치 파일들을 업로드 중입니다. 시간이 다소 소요될 수 있습니다...")
    api.upload_folder(
        folder_path=weights_dir,
        repo_id=repo_id,
        path_in_repo="weights",
        commit_message="Update model weights after training"
    )
    print(f"모든 가중치가 성공적으로 '{repo_id}'에 업로드되었습니다!")

if __name__ == "__main__":
    upload_weights_to_hf()
