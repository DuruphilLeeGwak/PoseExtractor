# Ghost Filter v5.1 Improvement Summary

## Analysis of Current Problems

### Problem 1: Ankle/Foot Removal
- **Issue**: `boundary_tolerance=30px` is too strict
- **Effect**: Normal feet at image bottom are marked as "dummy"
- **Symptoms**: In full-body photos, ankles/feet disappear even when clearly visible

### Problem 2: False Occlusion for Clear Arms/Hands
- **Issue**: 
  - `hand_min_avg_confidence=3.0` is too high
  - `hand_min_distance_std=50.0` filters out small/distant hands
- **Effect**: Even clearly visible arms/hands are marked as occluded (semi-transparent)

### Problem 3: Occlusion Misjudgment
- **Issue**: Same criteria applied to all body parts
- **Effect**: High-confidence keypoints still fail geometric checks

---

## Improvements Implemented

### 1. Dynamic Boundary Tolerance
**Before**:
- `boundary_tolerance: 30.0` (uniform for all edges)

**After**:
- **Default**: `10px` (top/left/right edges)
- **Bottom edge**: `80px` tolerance + requires `confidence >= 4.0`
- **Result**: Normal feet (high confidence) preserved, dummies (low confidence) removed

### 2. Body Part-Specific Occlusion Criteria

#### Hands (LHand/RHand)
- `min_avg_confidence`: 3.0 → **2.0** ✅
- `min_distance_std`: 50.0 → **30.0** ✅
- **Bypass**: Skip geometric checks if `avg_conf > 3.5`
- **Effect**: Small/distant hands with avg_conf 2.0-3.0 now pass

#### Face
- `min_avg_confidence`: **4.0** (small area, typically high confidence)
- `min_distance_std`: **15.0**
- **Bypass**: Skip all checks if `avg_conf > 5.0`

#### Arms/Legs (LArm/RArm/LLeg/RLeg)
- **Skip** `dist_std` check (long body parts)
- **Skip** `near_ratio` and `far_count` checks
- `min_avg_confidence`: **1.5** (low confidence is normal)

#### Feet (LFoot/RFoot)
- **Skip** all geometric checks (bottom location is normal)
- `min_avg_confidence`: **1.5**

---

## Expected Results

### ✅ Full-Body Photos
- Ankles/feet at image bottom render normally
- Feet with confidence ≥ 4.0 allowed within 80px of bottom edge

### ✅ Arms/Hands
- Clearly visible arms (high confidence) avoid false occlusion
- Small/distant hands with avg_conf 2.0-3.0 judged as normal

### ✅ Dummy/Hallucination Filtering
- Still removes low confidence + boundary coordinates
- Still detects estimation patterns (coordinate clustering)

---

## Configuration Changes

### `default.yaml`
```yaml
boundary_tolerance: 10.0         # 30.0 → 10.0
hand_min_avg_confidence: 2.0     # 3.0 → 2.0
hand_min_distance_std: 30.0      # 50.0 → 30.0
```

### `ghost_filter.py`
- Bottom boundary: 80px tolerance (conditional on confidence)
- Body part-specific criteria (hands/face/arms/legs/feet)
- Confidence-based geometric check bypass

---

## Testing Instructions

1. **Enable Ghost Filter**
   - Set `enabled: true` in `default.yaml`

2. **Test with Full-Body Photos**
   - Verify ankles/feet are rendered
   - Check `_ghostfilter_layers_debug.txt` for verdict

3. **Test with Clear Arms/Hands**
   - Verify arms/hands are not semi-transparent
   - Check for `verdict=KEEP` in debug log

4. **Review Debug Logs**
   - File: `_ghostfilter_layers_debug.txt`
   - Look for: `[BODY][Step3.5] verdict=KEEP/OCCLUDED`
   - Verify body parts match expected behavior

---

## Key Improvements Summary

| Improvement | Before | After | Effect |
|-------------|--------|-------|--------|
| boundary_tolerance | 30px | 10px (80px bottom) | Preserve feet at bottom |
| hand_min_avg_conf | 3.0 | 2.0 | Accept small/distant hands |
| hand_min_dist_std | 50.0 | 30.0 | Accept compact hands |
| Body part criteria | Uniform | Customized | Reduce false occlusions |
| Confidence bypass | None | >3.5 (hands), >5.0 (face) | Skip checks for high-conf |

---

**Status**: ✅ Improvements implemented and ready for testing
