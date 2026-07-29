# Deployment Fix Plan

## Issues Found
1. **Keras/TensorFlow version mismatch**: `caption_model.keras` saved with Keras 3.x but requirements pin TF 2.15 (Keras 2.x)
2. **Deprecated Streamlit param**: `use_container_width` → `width='stretch'` / `width='content'`

## Steps
- [x] 1. Analyze deployment logs and source code
- [x] 2. Get user approval on fix plan
- [ ] 3. Fix `requirements.txt` - unpin tensorflow/keras versions
- [ ] 4. Fix `app.py` - replace deprecated `use_container_width` with `width='stretch'`/`width='content'`
- [ ] 5. Push changes to GitHub
- [ ] 6. Verify deployment on Streamlit Community Cloud

