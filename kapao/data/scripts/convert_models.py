import torch
import pickle

import models.yolo
import models.experimental
import utils


device = utils.torch_utils.select_device(0, batch_size=1)
if device.type != "cuda":
    raise ValueError("Failed to load CUDA")
for model_name in ("kapao_s_coco", "kapao_m_coco", "kapao_l_coco", "kapao_s_crowdpose", "kapao_m_crowdpose", "kapao_l_crowdpose"):
    try:
        model = models.experimental.attempt_load(f"{model_name}.pt", map_location=device)
        meta = {key: val for key, val in vars(model).items() if not key.startswith("_")}
        torch.save(model.state_dict(), f"{model_name}.pth")
        with open(f"{model_name}.pkl", 'wb') as f:
            pickle.dump(meta, f)
    except Exception as exc:
        print(exc)
