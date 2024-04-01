
```
pip install -r requirements.txt
python -m playground.train with use_curriculum=True

python -m playground.enjoy --env Walker3DStepperEnv-v0 --net models/Walker3DStepperEnv-v0_best.pt --curriculum 0 --plank_class LargePlank
```