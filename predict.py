# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
#https://github.com/replicate/cog/blob/main/docs/getting-started-own-model.md

from typing import List
import subprocess
import os

from cog import BasePredictor, Input, Path
from foleycrafter.models.onset import torch_utils
from foleycrafter.utils.util import build_foleycrafter
from foleycrafter.pipelines.auffusion_pipeline import Generator
from foleycrafter.models.time_detector.model import VideoOnsetNet
from inference import run_inference
import torch
#nprompt optional

class Config:
    def __init__(self):
        '''
        config = {"prompt":prompt,"nprompt":"","seed":42,"semantic_scale":1.0,"temporal_scale":0.2,"input":str(video),"ckpt":"checkpoints/","save_dir":"output/","pretrain":"auffusion/auffusion-full-no-adapter","device":"cuda"}
        '''
        self.prompt = ""
        self.nprompt = ""
        self.seed = 42
        self.semantic_scale = 1.0
        self.temporal_scale = 0.2
        self.input = ''
        self.ckpt = 'checkpoints/'
        self.save_dir = 'output/'
        self.pretrain = 'auffusion/auffusion-full-no-adapter'
        self.device = 'cuda'
    
    

    
    

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        print("setup2")
        
        global pipe, vocoder, time_detector
        config = Config()

        # Load FoleyCrafter pipe
        pipe = build_foleycrafter().to(config.device)
        
        # Load temporal adapter
        temporal_ckpt_path = os.path.join(config.ckpt, "temporal_adapter.ckpt")
        ckpt = torch.load(temporal_ckpt_path)
        
        # Process and load the temporal adapter weights
        if "state_dict" in ckpt.keys():
            ckpt = ckpt["state_dict"]
        load_gligen_ckpt = {}
        for key, value in ckpt.items():
            if key.startswith("module."):
                load_gligen_ckpt[key[len("module."):]] = value
            else:
                load_gligen_ckpt[key] = value
        m, u = pipe.controlnet.load_state_dict(load_gligen_ckpt, strict=False)
        print(f"### Control Net missing keys: {len(m)}; \n### unexpected keys: {len(u)};")

        # Load semantic adapter
        pipe.load_ip_adapter(
            os.path.join(config.ckpt, "semantic"), 
            subfolder="", 
            weight_name="semantic_adapter.bin", 
            image_encoder_folder=None
        )
        ip_adapter_weight = config.semantic_scale
        pipe.set_ip_adapter_scale(ip_adapter_weight)

        # Load vocoder
        vocoder_config_path = config.ckpt
        vocoder = Generator.from_pretrained(vocoder_config_path, subfolder="vocoder").to(config.device)

        # Load time detector
        time_detector_ckpt = os.path.join(config.ckpt, "timestamp_detector.pth.tar")
        time_detector = VideoOnsetNet(False)
        time_detector, _ = torch_utils.load_model(time_detector_ckpt, time_detector, device=config.device, strict=True)
        
        

    def predict(
        self,
        video: Path = Input(description="Input video"),
        prompt: str = Input(description="Prompt to generate audio"),
        #nprompt with default value of ""
        nprompt: str = Input(description="Negative prompt for audio generation"),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        
        #get the config from above
        
        
        #run_inference(config, pipe, vocoder, time_detector):
        print("video",video)
        print("prompt",prompt)
        print("nprompt",nprompt)
        video_ext = video.split('.')[-1]
        output_video = video.replace('.' + video_ext, '_folycrafter.' + video_ext)
        out_audio = video.replace('.' + video_ext, '.wav')
        
        config = Config()
        #add the video to the config
        config.input = str(video)
        config.prompt = prompt
        config.nprompt = nprompt
        cwd = os.getcwd()
        #subprocess.call(["python","inference.py","--input",config.input,"--prompt",config.prompt,"--nprompt",config.nprompt],cwd=cwd)  
    
        out_audio,output_video = run_inference(config, pipe, vocoder, time_detector)
        #run_inference(config, pipe, vocoder, time_detector):
        return [Path(out_audio),Path(output_video)]
    