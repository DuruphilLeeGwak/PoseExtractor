"""
Pose Transfer API Module (v7.1 - Report Formatting Fix)

위치: pose_transfer/api.py
변경사항:
- [Fix] _create_report: 딕셔너리(Depth Stats 등)를 재귀적으로 출력하는 로직 추가
- [Fix] debug_report.txt에 누락되던 상세 정보 모두 기록
"""

# ==========================================
# 1. 라이브러리 임포트
# ==========================================
import sys          # 시스템 파라미터 및 함수 제어
import os           # 운영체제 상호작용 (환경변수, 파일 경로 등)
import yaml         # YAML 설정 파일 파싱
import shutil       # 파일 복사/이동 등 고수준 파일 연산
from pathlib import Path  # 객체 지향 파일 경로 처리
from datetime import datetime  # 현재 시간/날짜 처리
from typing import Dict, Union, Any  # 타입 힌팅용 모듈
import numpy as np  # 수치 연산 및 배열 처리

# Mac/Linux 환경에서 OpenMP 라이브러리 중복 로드 에러 방지 (Intel MKL 관련 이슈 해결)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ==========================================
# 2. 내부 모듈 임포트
# ==========================================
# 파이프라인 설정 클래스와 핵심 처리 로직 클래스 임포트
from .pipeline import PipelineConfig, PoseTransferPipeline
# 입출력 유틸리티 (JSON 저장, 이미지 저장, OpenPose 포맷 변환)
from .utils.io import save_json, save_image, convert_to_openpose_format
# 얼굴 전송 결과 시각화 도구
from .utils.face_transfer_visualizer import generate_face_transfer_image
# 전송 로직 설정 클래스
from .transfer import TransferConfig

class PoseTransferAPI:
    """
    포즈 이송(Pose Transfer) 전체 과정을 관장하는 상위 API 클래스.
    설정 로드, 파이프라인 초기화, 실행 및 결과 저장을 담당함.
    """

    def __init__(self, base_dir: str = None):
        """
        초기화 메서드
        :param base_dir: 프로젝트 루트 디렉토리 경로 (없으면 현재 파일 기준 상위로 자동 설정)
        """
        # base_dir이 주어지면 Path 객체로 변환, 없으면 현재 파일(api.py)의 상위 상위 폴더를 루트로 지정
        if base_dir: self.base_dir = Path(base_dir)
        else: self.base_dir = Path(__file__).parent.parent
        
        # 기본 설정 파일 경로 지정 (pose_transfer/config/default.yaml)
        self.config_path = self.base_dir / "pose_transfer" / "config" / "default.yaml"
        
        # 설정 파일 로드 및 내부 변수 설정
        self._load_config()
        
        print(f"🚀 Initializing Pose Transfer Pipeline...")
        # 로드된 설정을 바탕으로 핵심 파이프라인(PoseTransferPipeline) 인스턴스 생성
        self.pipeline = PoseTransferPipeline(self.pipeline_config, self.transfer_config)
        print("✅ Pipeline Ready.")

    def _load_config(self):
        """
        YAML 설정 파일을 읽어와 파이프라인 및 출력 설정에 적용하는 메서드
        """
        yaml_conf = {} # 설정 내용을 담을 딕셔너리 초기화
        print(f"\n🔍 [Config Check]")
        
        # 설정 파일이 존재하는지 확인
        if self.config_path.exists():
            try:
                # UTF-8 인코딩으로 YAML 파일 읽기
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    yaml_conf = yaml.safe_load(f) or {} # 파일이 비었으면 빈 딕셔너리 반환
                print(f"   ✅ Loaded: {self.config_path}")
            except: pass # 읽기 실패 시 예외 무시 (기본값 사용)
        
        # --- 출력(Output) 관련 설정 매핑 ---
        out_conf = yaml_conf.get('output', {})  # YAML의 'output' 섹션 가져오기
        dbg_conf = out_conf.get('debug', {})    # output 내 'debug' 섹션 가져오기
        
        # API 실행 시 저장할 항목들을 결정하는 플래그 설정
        self.output_config = {
            'save_keypoints': out_conf.get('save_json', True),       # 결과 JSON 저장 여부
            'save_skeleton': out_conf.get('save_skeleton_image', True), # 뼈대 이미지 저장 여부
            'save_debug_image': dbg_conf.get('save_bbox', True),     # 바운딩 박스 디버그 이미지 저장 여부
            'save_source_modified': dbg_conf.get('save_overlay', True), # 수정된 원본 오버레이 저장 여부
            'save_report': dbg_conf.get('save_text', True),          # 텍스트 리포트 저장 여부
            'save_face_debug': dbg_conf.get('save_face_viz', False), # 얼굴 전송 디버그 이미지 저장 여부
            'save_depth': dbg_conf.get('save_depth', False)          # 깊이 지도(Depth Map) 저장 여부
        }
        
        # --- 파이프라인(Pipeline) 설정을 위한 평탄화(Flattening) ---
        p_flat = {}
        rend_conf = yaml_conf.get('rendering', {}) # 렌더링 설정
        p_flat['point_radius'] = rend_conf.get('point_radius', 4)       # 키포인트 점 크기
        p_flat['line_thickness'] = rend_conf.get('line_thickness', 4)   # 뼈대 선 두께
        
        cf_conf = yaml_conf.get('cross_filter', {}) # 교차 필터 설정
        p_flat['cross_filter_enabled'] = cf_conf.get('enabled', True)   # 필터 사용 여부
        
        d_conf = yaml_conf.get('depth_anything', {}) # Depth Estimation 모델 설정
        p_flat['depth_enabled'] = d_conf.get('enabled', False)          # Depth 기능 켜기/끄기
        p_flat['depth_model_type'] = d_conf.get('model', 'depth_anything_v2_vitl') # 사용할 모델 버전
        
        # 자동 크롭 및 캔버스 패딩 설정
        p_flat['auto_crop_enabled'] = out_conf.get('auto_crop_enabled', False)
        p_flat['canvas_padding_ratio'] = out_conf.get('canvas_padding_ratio', 0.0)
        # 디버그 바운딩 박스 시각화 여부는 output config와 동기화
        p_flat['debug_bbox_visualization'] = self.output_config['save_debug_image']
        
        # 설정 상태 로깅
        print(f"   🎨 Rendering: Radius={p_flat['point_radius']}, Thick={p_flat['line_thickness']}")
        print(f"   🧭 Depth Enabled: {p_flat['depth_enabled']} (Model: {p_flat['depth_model_type']})")

        # 딕셔너리를 설정 객체(Config Object)로 변환하여 저장
        self.pipeline_config = PipelineConfig.from_dict(p_flat)
        self.transfer_config = TransferConfig.from_dict(yaml_conf.get('transfer', {}))

    def execute(self, source_path, reference_path, output_dir, prefix="trans"):
        """
        실제 포즈 이송 작업을 수행하는 메인 함수
        :param source_path: 원본 이미지 경로 (Person A)
        :param reference_path: 참조 이미지 경로 (Person B - Target Pose)
        :param output_dir: 결과물을 저장할 폴더 경로
        :param prefix: 결과 파일명 접두사 (기본값: "trans")
        """
        # 입력받은 경로들을 Path 객체로 변환
        src_p = Path(source_path)
        ref_p = Path(reference_path)
        out_d = Path(output_dir)
        
        # 결과물을 저장할 하위 디렉토리 정의
        dir_trans = out_d / "trans"  # 최종 결과물
        dir_src = out_d / "src"      # 원본 처리 결과
        dir_ref = out_d / "ref"      # 참조 처리 결과
        
        # 디렉토리 생성 (parents=True: 상위 폴더 없으면 생성, exist_ok=True: 이미 있어도 에러 안 냄)
        for d in [out_d, dir_trans, dir_src, dir_ref]: d.mkdir(parents=True, exist_ok=True)
            
        print(f"\n[API] Running Transfer: {src_p.name} -> {ref_p.name}")
        
        # ==========================================
        # [CORE] 파이프라인 실행 (포즈 추출 및 이송)
        # ==========================================
        result = self.pipeline.transfer(src_p, ref_p)
        
        # 1. Source(원본) 관련 데이터 저장
        try: shutil.copy2(src_p, dir_src / src_p.name) # 원본 이미지 복사
        except: pass
        
        if self.output_config['save_skeleton']:
            # 원본 이미지 로드 및 뼈대 시각화
            src_img_tmp = self.pipeline.canvas_mgr.load_image_safe(src_p)
            h, w = src_img_tmp.shape[:2]
            # 검은 배경에 뼈대만 그리기
            src_sk = self.pipeline.renderer.render_skeleton_only((h, w, 3), result.source_keypoints, result.source_scores)
            save_image(src_sk, dir_src / "src_sk.jpg")
            # 원본 이미지 위에 뼈대 오버레이
            src_ov = self.pipeline.renderer.render(src_img_tmp, result.source_keypoints, result.source_scores)
            save_image(src_ov, dir_src / "src_rend.jpg")
            
        if self.output_config['save_debug_image'] and result.src_debug_image is not None:
            # 바운딩 박스 등이 그려진 디버그 이미지 저장
            save_image(result.src_debug_image, dir_src / "src_debug_bbox.jpg")
            
        if self.output_config['save_keypoints']:
            # 키포인트 데이터를 OpenPose 호환 JSON 포맷으로 변환 후 저장
            src_json = convert_to_openpose_format(result.source_keypoints[None,...], result.source_scores[None,...], (h, w))
            save_json(src_json, dir_src / "src_kp.json")
            # 감지된 텍스트 정보 저장 (신뢰도 점수 등)
            self._save_debug_txt(dir_src / "src_debug.txt", result.src_debug_text)
            
        if self.output_config['save_depth'] and result.src_depth_map is not None:
            # Depth Map 저장
            save_image(result.src_depth_map, dir_src / "src_depth.jpg")

        # 2. Reference(참조) 관련 데이터 저장 (로직은 Source와 동일)
        try: shutil.copy2(ref_p, dir_ref / ref_p.name)
        except: pass
        
        if self.output_config['save_skeleton']:
            ref_img_tmp = self.pipeline.canvas_mgr.load_image_safe(ref_p)
            h, w = ref_img_tmp.shape[:2]
            ref_ov = self.pipeline.renderer.render(ref_img_tmp, result.reference_keypoints, result.reference_scores)
            save_image(ref_ov, dir_ref / "ref_rend.jpg")
            ref_sk = self.pipeline.renderer.render_skeleton_only((h, w, 3), result.reference_keypoints, result.reference_scores)
            save_image(ref_sk, dir_ref / "ref_sk.jpg")
            
        if self.output_config['save_debug_image'] and result.ref_debug_image is not None:
            save_image(result.ref_debug_image, dir_ref / "ref_debug_bbox.jpg")
            
        if self.output_config['save_keypoints']:
            ref_json = convert_to_openpose_format(result.reference_keypoints[None,...], result.reference_scores[None,...], (h, w))
            save_json(ref_json, dir_ref / "ref_kp.json")
            self._save_debug_txt(dir_ref / "ref_debug.txt", result.ref_debug_text)
            
        if self.output_config['save_depth'] and result.ref_depth_map is not None:
            save_image(result.ref_depth_map, dir_ref / "ref_depth.jpg")

        # 3. Transferred(결과) 데이터 저장
        if self.output_config['save_skeleton']:
            # 이송된 포즈(뼈대) 이미지 저장
            save_image(result.skeleton_image, dir_trans / f"{prefix}_sk.jpg")
            # 수정된 원본 이미지(modified_source) 위에 이송된 포즈 오버레이
            trans_ov = self.pipeline.renderer.render(result.modified_source_image, result.transferred_keypoints, result.transferred_scores)
            save_image(trans_ov, dir_trans / f"{prefix}_rend.jpg")
            
        if self.output_config['save_source_modified']:
            # 포즈에 맞춰 변형된 원본 이미지 저장
            save_image(result.modified_source_image, dir_trans / f"{prefix}_src_mod.jpg")
            
        if self.output_config['save_keypoints']:
            # 결과 키포인트 JSON 저장
            save_json(result.to_json(), dir_trans / f"{prefix}_kp.json")
            
        if self.output_config['save_report']:
            # [중요] 상세 리포트 생성 및 저장 (재귀 로직 포함된 함수 호출)
            rpt = self._create_report(result)
            with open(dir_trans / "debug_report.txt", "w", encoding="utf-8") as f:
                f.write(rpt)
                
        if self.output_config['save_face_debug']:
            # 얼굴 이송 과정 시각화 이미지 생성 및 저장
            face_vis = generate_face_transfer_image(result.processing_info.get('transfer_log', {}), result.source_keypoints, result.source_scores, result.reference_keypoints, result.reference_scores, result.transferred_keypoints, result.transferred_scores)
            if face_vis is not None: save_image(face_vis, dir_trans / "face_debug.jpg")

        print(f"[API] Process Finished. Output saved to {out_d}")
        return {} # 빈 딕셔너리 반환 (필요 시 결과 메타데이터 반환 가능)

    def _save_debug_txt(self, path, content):
        """텍스트 내용을 파일로 저장하는 헬퍼 함수"""
        if not content: return # 내용 없으면 스킵
        with open(path, "w", encoding="utf-8") as f: f.write(content)

    def _create_report(self, result):
        """
        전송 결과에 대한 상세 리포트를 텍스트 형식으로 생성
        :param result: 파이프라인 처리 결과 객체
        :return: 리포트 문자열
        """
        lines = [f"Pose Transfer Report - {datetime.now()}"]
        lines.append("-" * 50)
        
        # 정렬(Alignment) 정보 기록 (스케일, 오프셋 등)
        if result.alignment_info:
            ai = result.alignment_info
            lines.append(f"[Layout]")
            lines.append(f"  Strategy: {ai.anchor_type}")
            lines.append(f"  Scale   : {ai.global_scale:.3f}")
            lines.append(f"  Offset  : {ai.offset_vector.astype(int)}")
            lines.append(f"  Anchor(S): {ai.anchor_point_src}")
        
        lines.append("\n" + "="*50)
        lines.append("[5] Transfer Processing Log")
        lines.append("="*50)
        
        # 전송 처리 로그 가져오기
        log = result.processing_info.get('transfer_log', {})
        
        # [Fix] 중첩된 딕셔너리를 보기 좋게 출력하기 위한 내부 재귀 함수
        def _print_dict(d, indent=0):
            res = []
            pad = "  " * indent # 들여쓰기 계산
            for k, v in d.items():
                if isinstance(v, dict):
                    # 값이 딕셔너리인 경우: 키 출력 후 재귀 호출로 내부 진입
                    res.append(f"{pad}{k}:")
                    res.extend(_print_dict(v, indent+1))
                else:
                    # 일반 값인 경우: 키와 값을 바로 출력
                    res.append(f"{pad}{k}: {v}")
            return res

        # 재귀 함수를 사용하여 로그 전체를 리스트로 변환 후 추가
        lines.extend(_print_dict(log))
        return "\n".join(lines) # 줄바꿈 문자로 합쳐서 반환
