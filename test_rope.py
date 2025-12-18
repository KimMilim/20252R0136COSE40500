import torch
from vit import VisionTransformer, precompute_freqs_cis, apply_rotary_emb

def test_rope_helper_functions():
    print("\n[Test 1] RoPE 헬퍼 함수 단위 테스트 (Math & Shape Check)")
    
    # 설정: Batch=2, Seq=65, Head=3, Dim=64
    B, N, num_heads, head_dim = 2, 65, 3, 64
    
    # 1. 가상의 Query, Key 생성
    q = torch.randn(B, N, num_heads, head_dim)
    k = torch.randn(B, N, num_heads, head_dim)
    
    # 2. 주파수(각도) 계산 테스트
    freqs_cis = precompute_freqs_cis(head_dim, N)
    print(f"  - precompute_freqs_cis shape: {freqs_cis.shape}")
    
    # 검증: freqs_cis는 (Seq_len, head_dim/2) 여야 함
    expected_freq_shape = (N, head_dim // 2)
    assert freqs_cis.shape == expected_freq_shape, f"Shape Mismatch! Expected {expected_freq_shape}, got {freqs_cis.shape}"
    print("  - [PASS] Freqs shape is correct.")

    # 3. 회전 적용 테스트
    # broadcasting을 위해 freqs_cis의 차원을 맞춰서 넣어보는 시뮬레이션
    freqs_cis_input = freqs_cis[:N] # 시퀀스 길이만큼 자르기
    
    try:
        q_rot, k_rot = apply_rotary_emb(q, k, freqs_cis_input)
        print(f"  - Rotated Q shape: {q_rot.shape}")
        print("  - [PASS] apply_rotary_emb executed without error.")
    except Exception as e:
        print(f"  - [FAIL] Error inside apply_rotary_emb: {e}")
        return False
        
    return True

def test_vit_integration():
    print("\n[Test 2] ViT 모델 통합 테스트 (Integration Check)")
    
    # 1. Baseline ViT 생성 (use_rope=False)
    model_base = VisionTransformer(use_rope=False)
    if model_base.pos_embed is not None and model_base.freqs_cis is None:
        print("  - [PASS] Baseline Model: pos_embed is ON, freqs_cis is OFF.")
    else:
        print("  - [FAIL] Baseline Model initialization wrong.")

    # 2. RoPE ViT 생성 (use_rope=True)
    model_rope = VisionTransformer(use_rope=True)
    
    # 검증: RoPE 모델은 pos_embed가 없어야 하고, freqs_cis가 있어야 함
    if model_rope.pos_embed is None:
        print("  - [PASS] RoPE Model: Standard pos_embed is correctly DISABLED (None).")
    else:
        print(f"  - [FAIL] RoPE Model: pos_embed should be None, but got {type(model_rope.pos_embed)}.")

    if model_rope.freqs_cis is not None:
        print("  - [PASS] RoPE Model: freqs_cis is initialized.")
    else:
        print("  - [FAIL] RoPE Model: freqs_cis is Missing!")

    # 3. Forward Pass 테스트
    dummy_img = torch.randn(2, 3, 32, 32)
    
    try:
        out = model_rope(dummy_img)
        print(f"  - Output Shape: {out.shape}")
        if out.shape == (2, 10):
            print("  - [PASS] RoPE Forward Pass successful.")
        else:
            print("  - [FAIL] Output shape mismatch.")
    except Exception as e:
        print(f"  - [FAIL] Runtime Error during Forward Pass: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_rope_helper_functions()
    test_vit_integration()
    print("\n=== 모든 테스트 완료 ===")