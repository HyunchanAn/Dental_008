import streamlit as st
import streamlit.components.v1 as components
import os
import glob
import json
from PIL import Image

st.set_page_config(page_title="DENTEX 무치악 판독 UI", layout="centered")

@st.cache_data
def load_image_paths():
    base_dir = os.path.expanduser('~/.cache/dentex_dataset/yolo_format/images')
    train_imgs = glob.glob(os.path.join(base_dir, 'train', '*.jpg')) + glob.glob(os.path.join(base_dir, 'train', '*.png'))
    val_imgs = glob.glob(os.path.join(base_dir, 'val', '*.jpg')) + glob.glob(os.path.join(base_dir, 'val', '*.png'))
    
    # 제외: 우리가 프로그램으로 임의 생성했던 hardneg 조각 이미지들은 제외
    all_imgs = [p for p in (train_imgs + val_imgs) if 'hardneg_' not in os.path.basename(p)]
    return sorted(all_imgs)

def load_labels():
    label_file = os.path.expanduser('~/.cache/dentex_dataset/manual_labels.json')
    if os.path.exists(label_file):
        with open(label_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_labels(labels):
    label_file = os.path.expanduser('~/.cache/dentex_dataset/manual_labels.json')
    with open(label_file, 'w', encoding='utf-8') as f:
        json.dump(labels, f, indent=4, ensure_ascii=False)

def main():
    st.title("🦷 파노라마 무치악(Edentulous) 수동 분류기")
    st.markdown("""
    **사용 가이드**: 화면에 나타나는 엑스레이를 보고 치아가 아예 없는 '무치악' 환자인지, 
    정상 환자인지 판별해 주세요.
    - **O (무치악)**: 단축키 `3` - 치아가 하나도 없음 (추후 라벨 삭제)
    - **X (정상)**: 단축키 `2` - 치아가 1개라도 있음 (유지)
    - **이전 이미지**: 단축키 `1`
    """)

    # 키보드 단축키 JS 인젝션 (중복 리스너 방지)
    components.html(
        """
        <script>
        const doc = window.parent.document;
        if (!doc.getElementById('keyboard_shortcuts')) {
            const script = doc.createElement('script');
            script.id = 'keyboard_shortcuts';
            script.innerHTML = `
                document.addEventListener('keydown', function(e) {
                    if (['1', '2', '3'].includes(e.key)) {
                        const btns = Array.from(document.querySelectorAll('button'));
                        let btn = null;
                        if (e.key === '1') btn = btns.find(b => b.innerText.includes('이전 이미지'));
                        if (e.key === '2') btn = btns.find(b => b.innerText.includes('정상'));
                        if (e.key === '3') btn = btns.find(b => b.innerText.includes('무치악'));
                        if(btn) btn.click();
                    }
                });
            `;
            doc.head.appendChild(script);
        }
        </script>
        """,
        height=0,
        width=0,
    )

    image_paths = load_image_paths()
    total_imgs = len(image_paths)
    
    if total_imgs == 0:
        st.error("이미지를 찾을 수 없습니다. YOLO 데이터셋 경로를 확인하세요.")
        return

    # 상태 초기화
    if 'labels' not in st.session_state:
        st.session_state.labels = load_labels()
    
    # 안 한 것부터 띄우기 위해 인덱스 탐색
    if 'current_idx' not in st.session_state:
        start_idx = 0
        for i, path in enumerate(image_paths):
            if path not in st.session_state.labels:
                start_idx = i
                break
        st.session_state.current_idx = start_idx

    idx = st.session_state.current_idx
    if idx >= total_imgs:
        st.success("🎉 모든 이미지의 라벨링이 완료되었습니다! 창을 닫고 AI에게 스크립트 실행을 지시해 주세요.")
        st.progress(1.0)
        return

    current_image_path = image_paths[idx]
    
    st.progress(idx / total_imgs)
    st.write(f"**진행 상황**: {idx + 1} / {total_imgs} (현재 파일: {os.path.basename(current_image_path)})")
    
    # 이전 라벨이 있다면 표시
    prev_label = st.session_state.labels.get(current_image_path, "없음")
    st.info(f"현재 저장된 판독 결과: **{prev_label}**")

    # 이미지 출력
    try:
        img = Image.open(current_image_path)
        st.image(img, use_container_width=True)
    except Exception as e:
        st.error(f"이미지 로드 실패: {e}")

    # 버튼 레이아웃
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("⬅️ 이전 이미지", use_container_width=True):
            if st.session_state.current_idx > 0:
                st.session_state.current_idx -= 1
                st.rerun()
                
    with col2:
        if st.button("❌ 정상 (치아 있음)", use_container_width=True):
            st.session_state.labels[current_image_path] = "X"
            save_labels(st.session_state.labels)
            st.session_state.current_idx += 1
            st.rerun()
            
    with col3:
        if st.button("⭕ 무치악 (치아 없음)", use_container_width=True, type="primary"):
            st.session_state.labels[current_image_path] = "O"
            save_labels(st.session_state.labels)
            st.session_state.current_idx += 1
            st.rerun()

if __name__ == "__main__":
    main()
