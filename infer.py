import torch
import struct
import numpy as np
from einops import rearrange
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from huggingface_hub import hf_hub_download

import tgs
from tgs.utils.typing import *
from tgs.utils.saving import SaverMixin
from tgs.utils.ops import points_projection
from tgs.data import CustomImageOrbitDataset
from tgs.utils.config import parse_structured
from tgs.utils.misc import load_module_weights
from tgs.utils.misc import todevice, get_device
from tgs.models.image_feature import ImageFeature
from tgs.utils.config import ExperimentConfig, load_config

class TGS(torch.nn.Module, SaverMixin):
    @dataclass
    class Config:
        weights: Optional[str] = None
        weights_ignore_modules: Optional[List[str]] = None

        camera_embedder_cls: str = ""
        camera_embedder: dict = field(default_factory=dict)

        image_feature: dict = field(default_factory=dict)

        image_tokenizer_cls: str = ""
        image_tokenizer: dict = field(default_factory=dict)

        tokenizer_cls: str = ""
        tokenizer: dict = field(default_factory=dict)

        backbone_cls: str = ""
        backbone: dict = field(default_factory=dict)

        post_processor_cls: str = ""
        post_processor: dict = field(default_factory=dict)

        renderer_cls: str = ""
        renderer: dict = field(default_factory=dict)

        pointcloud_generator_cls: str = ""
        pointcloud_generator: dict = field(default_factory=dict)

        pointcloud_encoder_cls: str = ""
        pointcloud_encoder: dict = field(default_factory=dict)
    
    cfg: Config

    def load_weights(self, weights: str, ignore_modules: Optional[List[str]] = None):
        state_dict = load_module_weights(
            weights, ignore_modules=ignore_modules, map_location="cpu"
        )
        self.load_state_dict(state_dict, strict=False)

    def __init__(self, cfg):
        super().__init__()
        self.cfg = parse_structured(self.Config, cfg)
        self._save_dir: Optional[str] = None

        self.image_tokenizer = tgs.find(self.cfg.image_tokenizer_cls)(
            self.cfg.image_tokenizer
        )

        assert self.cfg.camera_embedder_cls == 'tgs.models.networks.MLP'
        weights = self.cfg.camera_embedder.pop("weights") if "weights" in self.cfg.camera_embedder else None
        self.camera_embedder = tgs.find(self.cfg.camera_embedder_cls)(**self.cfg.camera_embedder)
        if weights:
            from tgs.utils.misc import load_module_weights
            weights_path, module_name = weights.split(":")
            state_dict = load_module_weights(
                weights_path, module_name=module_name, map_location="cpu"
            )
            self.camera_embedder.load_state_dict(state_dict)

        self.image_feature = ImageFeature(self.cfg.image_feature)

        self.tokenizer = tgs.find(self.cfg.tokenizer_cls)(self.cfg.tokenizer)

        self.backbone = tgs.find(self.cfg.backbone_cls)(self.cfg.backbone)

        self.post_processor = tgs.find(self.cfg.post_processor_cls)(
            self.cfg.post_processor
        )

        self.renderer = tgs.find(self.cfg.renderer_cls)(self.cfg.renderer)

        # pointcloud generator
        self.pointcloud_generator = tgs.find(self.cfg.pointcloud_generator_cls)(self.cfg.pointcloud_generator)

        self.point_encoder = tgs.find(self.cfg.pointcloud_encoder_cls)(self.cfg.pointcloud_encoder)

        # load checkpoint
        if self.cfg.weights is not None:
            self.load_weights(self.cfg.weights, self.cfg.weights_ignore_modules)
    
    def gaussian_model_to_ply_bytes(self, gaussian_model):
        # Get all properties
        xyz = gaussian_model.xyz
        rotation = gaussian_model.rotation
        scaling = gaussian_model.scaling
        opacity = gaussian_model.opacity
        rgb = (gaussian_model.shs[:, 0] + 1) / 2 * 255  # Convert SH to RGB
        
        num_points = len(xyz)
        
        header = [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {num_points}",
            "property float x",
            "property float y", 
            "property float z",
            "property float rot_w",
            "property float rot_x",
            "property float rot_y",
            "property float rot_z",
            "property float scale_x",
            "property float scale_y",
            "property float scale_z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "property float opacity",
            "end_header\n"
        ]
        
        # Move everything to CPU and convert to numpy for struct packing
        xyz = xyz.detach().cpu().numpy()
        rotation = rotation.detach().cpu().numpy()
        scaling = scaling.detach().cpu().numpy()
        opacity = opacity.detach().cpu().numpy()
        rgb = rgb.detach().cpu().numpy().astype(np.uint8)
        

        vertex_data = bytearray()
        for i in range(num_points):
            vertex_data.extend(struct.pack('fff', *xyz[i]))           # position
            vertex_data.extend(struct.pack('ffff', *rotation[i]))     # rotation
            vertex_data.extend(struct.pack('fff', *scaling[i]))       # scaling
            vertex_data.extend(struct.pack('BBB', *rgb[i]))          # color
            vertex_data.extend(struct.pack('f', float(opacity[i][0])))       # opacity
        
        return '\n'.join(header).encode('ascii') + vertex_data
    
    def _forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        # generate point cloud
        out = self.pointcloud_generator(batch)
        pointclouds = out["points"]

        batch_size, n_input_views = batch["rgb_cond"].shape[:2]

        # Camera modulation
        camera_extri = batch["c2w_cond"].view(*batch["c2w_cond"].shape[:-2], -1)
        camera_intri = batch["intrinsic_normed_cond"].view(*batch["intrinsic_normed_cond"].shape[:-2], -1)
        camera_feats = torch.cat([camera_intri, camera_extri], dim=-1)

        camera_feats = self.camera_embedder(camera_feats)

        input_image_tokens: Float[Tensor, "B Cit Nit"] = self.image_tokenizer(
            rearrange(batch["rgb_cond"], 'B Nv H W C -> B Nv C H W'),
            modulation_cond=camera_feats,
        )
        input_image_tokens = rearrange(input_image_tokens, 'B Nv C Nt -> B (Nv Nt) C', Nv=n_input_views)

        # get image features for projection
        image_features = self.image_feature(
            rgb = batch["rgb_cond"],
            mask = batch.get("mask_cond", None),
            feature = input_image_tokens
        )

        # only support number of input view is one
        c2w_cond = batch["c2w_cond"].squeeze(1)
        intrinsic_cond = batch["intrinsic_cond"].squeeze(1)
        proj_feats = points_projection(pointclouds, c2w_cond, intrinsic_cond, image_features)

        point_cond_embeddings = self.point_encoder(torch.cat([pointclouds, proj_feats], dim=-1))
        tokens: Float[Tensor, "B Ct Nt"] = self.tokenizer(batch_size, cond_embeddings=point_cond_embeddings)

        tokens = self.backbone(
            tokens,
            encoder_hidden_states=input_image_tokens,
            modulation_cond=None,
        )

        scene_codes = self.post_processor(self.tokenizer.detokenize(tokens))
        rend_out = self.renderer(scene_codes,
                                query_points=pointclouds,
                                additional_features=proj_feats,
                                **batch)

        return {**out, **rend_out}
    
    def forward(self, batch):
        out = self._forward(batch)
        batch_size = batch["index"].shape[0]
        for b in range(batch_size):
            if batch["view_index"][b, 0] == 0:
                plybytes = self.gaussian_model_to_ply_bytes(out["3dgs"][b])
                return plybytes
                
        
def run_triplane_gaussian_splatting(url, config_path="config.yaml", cam_dist=1.9, extras=None):
    device = get_device()
    print(f"Using device: {device}")

    cfg: ExperimentConfig = load_config(config_path, cli_args=extras)
    model_path = hf_hub_download(repo_id="VAST-AI/TriplaneGaussian", local_dir="./checkpoints", filename="model_lvis_rel.ckpt", repo_type="model")
    cfg.system.weights = model_path
    model = TGS(cfg=cfg.system).to(device)
    print("load model ckpt done.")

    cfg.data.image_list = [url]

    cfg.data.cond_camera_distance = cam_dist
    cfg.data.eval_camera_distance = cam_dist
    dataset = CustomImageOrbitDataset(cfg.data)
    dataloader = DataLoader(dataset,
                            batch_size=cfg.data.eval_batch_size, 
                            num_workers=cfg.data.num_workers,
                            shuffle=False,
                            collate_fn=dataset.collate)

    for batch in dataloader:
        batch = todevice(batch)
        print("Processing batch...")
        output = model(batch)
        with open('output.ply', 'wb') as f:
            f.write(output)

if __name__ == "__main__":
    run_triplane_gaussian_splatting()