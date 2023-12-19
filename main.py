import base64
import contextlib
import os
import uuid
import zipfile
from datetime import datetime
from io import BytesIO, StringIO

import PIL
import diffusers
import numpy
import requests
from diffusers import DiffusionPipeline, StableDiffusionXLPipeline, DPMSolverMultistepScheduler, AutoencoderTiny, \
    AutoPipelineForImage2Image
from diffusers.pipelines import BlipDiffusionPipeline
from diffusers import KDPM2AncestralDiscreteScheduler
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL, BlipDiffusionControlNetPipeline
import torch
import numpy as np
import cv2
from PIL import Image
from compel import Compel, ReturnedEmbeddingsType
from diffusers.models.attention_processor import AttnProcessor2_0
import torch
import os
from huggingface_hub import HfApi
from pathlib import Path
from diffusers.utils import load_image
from PIL import Image
import numpy as np
from transformers import pipeline
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

from tqdm.auto import tqdm
import signal
import requests
import urllib.request
import urllib.parse
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import re
import waitress

from flask import Flask, request, jsonify, render_template, url_for, send_from_directory, session, redirect, send_file
from flask_cors import CORS
from flask_session import Session
app = Flask(__name__)
CORS(app)
# Configura l'app per utilizzare il backend del filesystem per memorizzare i dati della sessione
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = True  # Imposta a True se desideri che le sessioni siano permanenti
app.config['SESSION_FILE_DIR'] = 'flask_session'  # Directory in cui salvare i dati della sessione

# Inizializza l'estensione Flask-Session
Session(app)
#set the secret key.  keep this really secret:
app.secret_key = 'A0Zr98j/3yX R~XHH!jmN]LWX/,?RT'

# Get file path
dir_path = os.path.dirname(os.path.realpath(__file__))

#Functions
def factorize(num: int, max_value: int) -> list[float]:
  result = []
  while num > max_value:
    result.append(max_value)
    num /= max_value
  result.append(round(num, 4))
  return result


from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)

#prompt related stuff
def prompt_worker_sdxl(prompt,negative_prompt,pipe_sdxl):
    compel = Compel(tokenizer=[pipe_sdxl.tokenizer, pipe_sdxl.tokenizer_2],
                        text_encoder=[pipe_sdxl.text_encoder, pipe_sdxl.text_encoder_2],
                        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                        requires_pooled=[False, True],
                    truncate_long_prompts=False)

    with torch.no_grad():
        conditioning, pooled = compel([prompt,negative_prompt])
    return conditioning, pooled

def prompt_worker(prompt,negative_prompt,pipeline):
    compel = Compel(tokenizer=pipeline.tokenizer, text_encoder=pipeline.text_encoder,truncate_long_prompts=False)
    with torch.no_grad():
        conditioning = compel.build_conditioning_tensor(prompt)
        negative_conditioning = compel.build_conditioning_tensor(negative_prompt)
        [conditioning, negative_conditioning] = compel.pad_conditioning_tensors_to_same_length([conditioning, negative_conditioning])

    return conditioning, negative_conditioning

#loaders
def load_tiny_ae(ae:str,device:str='cuda')->AutoencoderTiny:
    tiny_vae = AutoencoderTiny.from_pretrained(ae, torch_dtype=torch.float16).to(device)
    return tiny_vae


def make_pipeline_control(model:str,dtype:torch.dtype=torch.float16):
    pipeline = ControlNetModel.from_pretrained(model, torch_dtype=dtype)
    return pipeline


def make_pipeline_sdxl(model:str, pipe:str='default', scheduler_type:str="KDPM2D", dtype:torch.dtype=torch.float16, gpu_id:int=0,
                       is_karras=False):
    if '/' not in model:
        model = os.path.join(dir_path,'model',model)
    if pipe=='controlnet':
        pipeline = StableDiffusionXLControlNetPipeline.from_single_file(model,controlnet=controlnet_sdxl,torch_dtype=dtype,use_safetensors=True)
        pipeline.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)

    elif pipe=='standard':
        pipeline = StableDiffusionXLPipeline.from_single_file(model,torch_dtype=dtype,use_safetensors=True)
        pipeline.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)

    elif pipe=='img2img':
        pipeline = AutoPipelineForImage2Image.from_pretrained(model,torch_dtype=dtype,variant="fp16",use_safetensors=True,)

    else:
        raise NotImplementedError("Pipeline not implemented")
    # if the system is linux, compile the models
    #if os.name == 'posix' and (pipe == 'standard' or pipe == 'controlnet'):
     #   pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=True)
    if scheduler_type=='KDPM2DA':
        pipeline.scheduler = diffusers.KDPM2AncestralDiscreteScheduler(use_karras_sigmas=is_karras).from_config(pipeline.scheduler.config)
    elif scheduler_type=='KDPM2D':
        pipeline.scheduler = diffusers.KDPM2DiscreteScheduler(use_karras_sigmas=is_karras).from_config(pipeline.scheduler.config)
    elif scheduler_type=='DPM++':
        pipeline.scheduler =  diffusers.DPMSolverSDEScheduler(use_karras_sigmas=is_karras).from_config(pipeline.scheduler.config)
    elif scheduler_type=='Euler':
        pipeline.scheduler = diffusers.EulerDiscreteScheduler(use_karras_sigmas=is_karras).from_config(pipeline.scheduler.config)
    elif scheduler_type=='EulerA':
        pipeline.scheduler = diffusers.EulerAncestralDiscreteScheduler(use_karras_sigmas=is_karras).from_config(pipeline.scheduler.config)
    pipeline.enable_model_cpu_offload(gpu_id=gpu_id)
    pipeline.enable_vae_tiling()
    pipeline.enable_xformers_memory_efficient_attention()
    return pipeline


def make_translation_pipeline(model:str,len_in:str='ita_Latn',len_out='eng_Latn',device:str='cuda'):
    translator_tokenizer = AutoTokenizer.from_pretrained(model)
    translator_model = AutoModelForSeq2SeqLM.from_pretrained(model)
    translator = pipeline('translation', model=translator_model, tokenizer=translator_tokenizer, src_lang=len_in,
                          tgt_lang=len_out, device=device)
    return translator


#Functions latent related
def latents_to_pil(latents):
    '''
    Function to convert latents to images
    '''
    latents = (1 / 0.18215) * latents
    with torch.no_grad():
        sdxl_vae_preview.to(latents.dtype).to(latents.device)
        image = sdxl_vae_preview.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images


def callback(step, timestep, latents):
    global step_g, latent_g
    step_g+=5
    latent_g = latents


def traduci_in_inglese(testo, max_token=400):
    output = translator(testo, max_length=max_token)
    return output[0]['translation_text']



def download_file(
      link: str,
      path: str,
      block_size: int = 1024,
      force_download: bool = False,
      progress: bool = True,
      interrupt_check: bool = True
) -> str:

  def truncate_string(string: str, length: int):
    length -= 5 if length - 5 > 0 else 0
    curr_len = len(string)
    new_len = len(string[:length // 2] + "(...)" + string[-length // 2:])
    if new_len > curr_len:
      return string
    else:
      return string[:length // 2] + "(...)" + string[-length // 2:]

  def remove_char(string: str, chars: list):
    for char in chars:
      string = string.replace(char, "")
    return string

  # source: https://github.com/wkentaro/gdown/blob/main/gdown/download.py
  def google_drive_parse_url(url: str):
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)
    is_gdrive = parsed.hostname in ["drive.google.com", "docs.google.com"]
    is_download_link = parsed.path.endswith("/uc")

    if not is_gdrive:
      return is_gdrive, is_download_link

    file_id = None
    if "id" in query:
      file_ids = query["id"]
      if len(file_ids) == 1:
        file_id = file_ids[0]
    else:
      patterns = [r"^/file/d/(.*?)/view$", r"^/presentation/d/(.*?)/edit$"]
      for pattern in patterns:
        match = re.match(pattern, parsed.path)
        if match:
          file_id = match.groups()[0]
          break

    return file_id, is_download_link

  # source: https://github.com/wkentaro/gdown/blob/main/gdown/download.py
  def get_url_from_gdrive_confirmation(contents: str):
    url = ""
    for line in contents.splitlines():
      m = re.search(r'href="(/uc\?export=download[^"]+)', line)
      if m:
        url = "https://docs.google.com" + m.groups()[0]
        url = url.replace("&amp;", "&")
        break
      m = re.search('id="download-form" action="(.+?)"', line)
      if m:
        url = m.groups()[0]
        url = url.replace("&amp;", "&")
        break
      m = re.search('"downloadUrl":"([^"]+)', line)
      if m:
        url = m.groups()[0]
        url = url.replace("\\u003d", "=")
        url = url.replace("\\u0026", "&")
        break
      m = re.search('<p class="uc-error-subcaption">(.*)</p>', line)
      if m:
        error = m.groups()[0]
        raise RuntimeError(error)
    if not url:
      raise RuntimeError(
        "Cannot retrieve the link of the file. "
      )
    return url

  def interrupt(*args):
    if os.path.isfile(filepath):
      os.remove(filepath)
    raise KeyboardInterrupt

  # create folder if not exists
  if not os.path.exists(path):
    os.makedirs(path)

  # check if link is google drive link
  if not google_drive_parse_url(link)[0]:
    response = requests.get(link, stream=True, allow_redirects=True)
  else:
    if not google_drive_parse_url(link)[1]:
      # convert to direct link
      file_id = google_drive_parse_url(link)[0]
      link = f"https://drive.google.com/uc?id={file_id}"
    # test if redirect is needed
    response = requests.get(link, stream=True, allow_redirects=True)
    if response.headers.get("Content-Disposition") is None:
      page = urllib.request.urlopen(link)
      link = get_url_from_gdrive_confirmation(str(page.read()))
      response = requests.get(link, stream=True, allow_redirects=True)

  if response.status_code == 404:
    raise FileNotFoundError(f"File not found at {link}")

  # get filename
  content_disposition = response.headers.get("Content-Disposition")
  if content_disposition:
    filename = re.findall(r'filename=(.*?)(?:[;\n]|$)', content_disposition)[0]
  else:
    filename = os.path.basename(link)

  filename = remove_char(filename, ['/', '\\', ':', '*', '?', '"', "'", '<', '>', '|', ';'])
  filename = filename.replace(' ', '_')

  filepath = os.path.join(path, filename)

  # download file
  if os.path.isfile(filepath) and not force_download:
    print(f"{filename} already exists. Skipping download.")
  else:
    text = f"Downloading {truncate_string(filename, 50)}"
    with open(filepath, "wb") as file:
      total_size = int(response.headers.get("content-length", 0))
      with tqdm(
        total=total_size,
        unit="B",
        unit_scale=True,
        desc=text,
        unit_divisor=1024,
        disable=not progress,
      ) as pb:
        if interrupt_check:
          signal.signal(signal.SIGINT, lambda signum, frame: interrupt())
        for data in response.iter_content(block_size):
          pb.update(len(data))
          file.write(data)
  del response
  return filename


def upscale(
    img_list: list[PIL.Image.Image],
    model_name: str = "RealESRGAN_x4plus",
    scale_factor: float = 4,
    half_precision: bool = False,
    tile: int = 750,
    tile_pad: int = 10,
    pre_pad: int = 0,
) -> list[PIL.Image.Image]:
  global upsampler

  # check model
  if model_name == "RealESRGAN_x4plus":
    upscale_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    netscale = 4
    file_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
  elif model_name == "RealESRNet_x4plus":
    upscale_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    netscale = 4
    file_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth"
  elif model_name == "RealESRGAN_x4plus_anime_6B":
    upscale_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
    netscale = 4
    file_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
  elif model_name == "RealESRGAN_x2plus":
    upscale_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    netscale = 2
    file_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
  else:
    raise NotImplementedError("Model name not supported")
  # download model
  model_path = download_file(file_url, path=os.path.join("model","upscaler-model"), progress=True, interrupt_check=False)
  if upsampler is None:
      # declare the upscaler
      upsampler = RealESRGANer(
        scale=netscale,
        model_path=os.path.join("model","upscaler-model", model_path),
        dni_weight=None,
        model=upscale_model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=half_precision,
        gpu_id=None
      )

  # upscale
  torch.cuda.empty_cache()
  upscaled_imgs = []
  with tqdm(total=len(img_list)) as pb:
    for i, img in enumerate(img_list):
      img = numpy.array(img)
      outscale_list = factorize(scale_factor, netscale)
      with contextlib.redirect_stdout(StringIO()):
        for outscale in outscale_list:
          curr_img = upsampler.enhance(img, outscale=outscale)[0]
          img = curr_img
          torch.cuda.empty_cache()
        upscaled_imgs.append(Image.fromarray(img))

      pb.update(1)
  torch.cuda.empty_cache()


  return upscaled_imgs



#Segment the image to get a mask of the intrested area
def get_image_masks(images,object):
    from PIL import Image
    #from lang_sam import LangSAM
    image_dict = {}
    model = "model"#LangSAM(ckpt_path=os.path.join('model','4b8939.pth'))
    for image in images:
        risposta = requests.get(image)
        # Verifica che la richiesta sia andata a buon fine
        risposta.raise_for_status()
        # Crea un oggetto BytesIO a partire dai dati dell'immagine
        dati_immagine = BytesIO(risposta.content)
        # Apri l'immagine utilizzando Pillow
        image = Image.open(dati_immagine)
        #images are url to the local server
        masks, boxes, phrases, logits = model.predict(image, object)
        image_dict[image.filename] = masks.cpu().numpy().squeeze()
    return image_dict


def get_dense_image(image:PIL.Image.Image) -> PIL.Image.Image:
    depth_estimator = pipeline('depth-estimation')
    image = depth_estimator(image)['depth']
    image = np.array(image)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    control_image = Image.fromarray(image)
    return control_image

def get_canny_image(image:PIL.Image.Image) -> PIL.Image.Image:
    image = np.array(image)
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    control_image = Image.fromarray(image)
    return control_image


#Dirs
img_dir_sfondi = "sfondi_generati"
os.makedirs(img_dir_sfondi, exist_ok=True)
img_dir_immagini_finali = "immagini_generate"
os.makedirs(img_dir_immagini_finali, exist_ok=True)

#Constants
gpu_id = 1
step_g, timestep_g = 0, 0
inference_steps = 25
batch = 9
upsampler = None
tot_num_inference_steps = inference_steps*batch

#TinyVAEs
sdxl_vae_preview = load_tiny_ae("sayakpaul/taesdxl-diffusers")
vae_preview = load_tiny_ae("sayakpaul/taesd-diffusers")


#StableDiffusionXLPipelines
controlnet_conditioning_scale = 0.5
controlnet_sdxl = make_pipeline_control("diffusers/controlnet-canny-sdxl-1.0",dtype=torch.float16,)
vae_sdxl = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", torch_dtype=torch.float16, use_safetensors=True)
pipe_sdxl_cntrlnt = make_pipeline_sdxl("1_80.safetensors",pipe='controlnet',dtype=torch.float16)
pipe_sdxl = make_pipeline_sdxl("1_80.safetensors",pipe='standard',dtype=torch.float16)
#img2img_pipe_sdxl = make_pipeline_sdxl("1_80.safetensors",pipe='img2img',dtype=torch.float16)


#Translator
translator = make_translation_pipeline("facebook/nllb-200-distilled-600M")

def armonize(image:PIL.Image.Image,mask:PIL.Image.Image):
    from HDNet.inference import HdnetInference
    model = HdnetInference()
    immagini = model.elabortate_image(image,mask)
    return immagini

@app.route('/armonizza', methods=['GET','POST'])
def armonizza():
    #serve the harmonize html
    return render_template('harmonize_temp.html')

@app.route('/armonizza_posiziona_oggetto', methods=['GET','POST'])
def armonizza_posiziona_oggetto():
    #serve the harmonize html
    return render_template('harmonize_last.html')


@app.route('/rimuovi/sfondo', methods=['GET', 'POST'])
def serve_rimuovi_sfondo():
    return render_template('remove_background.html')

@app.route('/rimuovi/elaborato', methods=['GET','POST'])
def elabora_e_mostra():
    from rembg import remove
    from PIL import Image
    # read image from request
    image = request.files.get('file')
    # open the image with pil
    input = Image.open(image)
    output = remove(input)
    # save the generated images in immagini_generate/upscaled
    uninque_id = uuid.uuid4()
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    os.makedirs(os.path.join(img_dir_immagini_finali, 'immagini_senza_sfondo', dt_string), exist_ok=True)
    counter = 0
    counter += 1
    im_li = []
    output.save(os.path.join('immagini_generate','immagini_senza_sfondo',dt_string,str(counter) + '.png'))
    return send_file(os.path.join('immagini_generate','immagini_senza_sfondo',dt_string,str(counter) + '.png'), as_attachment=True)    #return jsonify({'redirect_url': im_li})


@app.route('/sfondi/<dirnew>/<filename>', methods=['GET','POST'])
def sfondi(filename, dirnew):
    return send_from_directory(os.path.join(img_dir_sfondi,dirnew), filename)


@app.route('/mostra/<dirnew>/<subdir>/<filename>', methods=['GET','POST'])
def mostra(filename,subdir, dirnew):
    return send_from_directory(os.path.join('immagini_generate',dirnew,subdir), filename)


@app.route('/mostra_immagini', methods=['GET','POST'])
def mostra_immagini():
    images = session.get('images')
    # Puoi passare variabili al tuo template qui, se necessario
    return render_template('grid_select.html', images=images)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/genera_background', methods=['GET','POST'])
def generate_bg_image():
    global step_g
    step_g = 0
    uninque_id = uuid.uuid4()
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    os.makedirs(os.path.join(img_dir_sfondi,dt_string), exist_ok=True)
    negative_prompt = 'text, watermark, glitch, overexposed, high-contrast, low quality, blurry,monochrome, people, furniture, distorted, jpeg-artifacts'
    data = request.form
    objec = data.get('object',"")
    #get the image from the request.
    image = request.files.get('image')
    if image is not None:
        #file storage to numpy array
        image = Image.open(image).convert('RGB')
        image = np.array(image)
        image = cv2.Canny(image, 100, 200)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = Image.fromarray(image)
    generator = torch.Generator(device="cuda")
    seed = generator.seed()
    generator.manual_seed(seed)
    #get the height and weight
    heigh = int(request.form.get('height'))
    weight = int(request.form.get('width'))

    # extract the object, the room, and the house style and the interior designer
    room = traduci_in_inglese("Stanza " + data['room'])
    house_style = traduci_in_inglese(data['style'])
    images= []
    if data['designer']:
        designer_name = traduci_in_inglese(data['designer'])
        prompt= f"a interior design (photograph)1.2 of an (empty):1.6 {room} floorsubsitutethis objectsubstitutethis inside a {house_style} wallsubstitutethis, sharp details, 4k, Interior design inspired by {designer_name}"
    else:
        prompt= f"a interior design (photograph)1.2 of an (empty):1.6 {room} floorsubsitutethis objectsubstitutethis inside a {house_style} wallsubstitutethis, sharp details, 4k"
    if data['object']:
        objec = traduci_in_inglese(data['object'])
        prompt_split = prompt.split('objectsubstitutethis')
        prompt = prompt_split[0] + f"with a {objec}" + prompt_split[1]
    if data['floor']:
        floor = traduci_in_inglese(data['floor'])
        prompt_split = prompt.split('floorsubsitutethis')
        prompt = prompt_split[0] + f"with a {floor} floor" + prompt_split[1]
    if data['wall']:
        wall = traduci_in_inglese(data['wall'])
        prompt_split = prompt.split('wallsubstitutethis')
        prompt = prompt_split[0] + f" with {wall} walls" + prompt_split[1]
    prompt = prompt.replace('floorsubsitutethis ', '').replace(' wallsubstitutethis', '').replace('objectsubstitutethis ', '')

    #create the conditionings
    conditioning, pooled = prompt_worker_sdxl(prompt, negative_prompt, pipe_sdxl_cntrlnt if image is not None else pipe_sdxl)
    if image is not None:

        for _ in range(batch):
            images.append(pipe_sdxl_cntrlnt(
                prompt_embeds=conditioning[0:1],
                pooled_prompt_embeds=pooled[0:1],
                negative_prompt_embeds=conditioning[1:2],
                negative_pooled_prompt_embeds=pooled[1:2],
                image=image,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                generator=generator,
                height=heigh,
                width=weight,
                num_inference_steps=inference_steps,
                callback=callback,
                callback_steps=5,
            ).images)
    else:
        for _ in range(batch):
            images.append(pipe_sdxl(
                prompt_embeds=conditioning[0:1],
                pooled_prompt_embeds=pooled[0:1],
                negative_prompt_embeds=conditioning[1:2],
                negative_pooled_prompt_embeds=pooled[1:2],
                generator=generator,
                height=heigh,
                width=weight,
                num_inference_steps=inference_steps,
                num_images_per_prompt=1,
                callback=callback,
                callback_steps=5,
            ).images)

    image_urls = []


    for image in images:
        index = str(images.index(image))
        image[0].save(os.path.join(img_dir_sfondi,dt_string,uninque_id.__str__()+"_"+index+"_"+".png" ))
        image_urls.append(url_for("sfondi",filename=str(uninque_id.__str__()+"_"+index+"_"+".png"),dirnew=dt_string))

    # Crea un buffer in memoria per il file zip
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for index, image_url in enumerate(image_urls, start=1):
                # Assicurati che il percorso dell'immagine sia corretto
                image_path = os.path.join('sfondi_generati',  dt_string, f"{uninque_id.__str__()}_{index}_.png")
                zip_file.write(image_path, arcname=f'{index}.png')

        # Imposta il puntatore del buffer all'inizio del file
        zip_buffer.seek(0)

    return send_file(
            zip_buffer,
            as_attachment=True,
            mimetype='application/zip',
            download_name=f'immagini_generate_{dt_string}.zip'
        )
        # return the images with send_file




@app.route('/current_step', methods=['GET','POST'])
def current_step():
    global step_g, latent_g
    percentuage = (step_g/tot_num_inference_steps)*100
    if percentuage == 100.0:
        decoded_img = latents_to_pil(latent_g.long())[0]
        buffered = BytesIO()
        decoded_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue())
        return jsonify({'progress':percentuage, 'image':img_str.decode('utf-8')})
    try:
        decoded_img = latents_to_pil(latent_g.long())[0]
        buffered = BytesIO()
        decoded_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue())
    except NameError:
        decoded_img = np.zeros((512,512,3))
        decoded_img = Image.fromarray(decoded_img.astype('uint8'))
        buffered = BytesIO()
        decoded_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue())

    return jsonify({'progress':percentuage, 'image':img_str.decode('utf-8')})


@app.route('/seleziona_immagini', methods=['GET', 'POST'])
def seleziona_immagini():
    # get the selected image list
    session['selected_images'] = request.values.get('image').split(',')
    return jsonify({'redirect_url': url_for('carica_soggetto')})


@app.route('/carica_soggetto', methods=['GET', 'POST'])
def carica_soggetto():
    return render_template('carica_soggetto.html')


@app.route('/upscale', methods=['GET', 'POST'])
def upscale_page():
    return render_template('upscale.html')


@app.route('/mostra_immagini_upscaled', methods=['GET', 'POST'])
def mostra_immagini_upscaled():
    images = session.get('images_upscaled')
    # Puoi passare variabili al tuo template qui, se necessario
    return render_template('grid_select_download.html', images=images)

@app.route('/upscale_download', methods=['GET', 'POST'])
def upscale_download():
    images =request.form.getlist('image')
    images=images[0].split(",")
    upscale_by=4
    risparmia_memoria = True
    image_list=[]
    for i in range(0,len(images)):
        img_link = images[i]
        risposta = requests.get(img_link)
        # aggiungi le imamigni pil a image list
        image_list.append(Image.open(BytesIO(risposta.content)))
    images_upscaled = upscale(image_list,tile=750,scale_factor=upscale_by,half_precision=risparmia_memoria, model_name="RealESRGAN_x2plus" if upscale_by == 2 else "RealESRGAN_x4plus")
    #save the generated images in immagini_generate/upscaled
    uninque_id = uuid.uuid4()
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    os.makedirs(os.path.join(img_dir_immagini_finali,'ingrandite',dt_string), exist_ok=True)
    counter=0
    im_li=[]
    for im in images_upscaled:
        counter+=1
        im.save('immagini_generate/ingrandite/'+dt_string+'/'+str(counter)+'.png')
        im_li.append(url_for('mostra',filename=str(counter)+'.png',dirnew='ingrandite',subdir=f'{dt_string}'))
    return jsonify({'redirect_url': im_li})


@app.route('/upscale_elaborate', methods=['GET', 'POST'])
def upscale_images():
    max_i = 0
    images = []
    #images are sent in request.file, adapt the code to get the images(variable numbero of images)

    for key in request.files.keys():
        if key.startswith('image_'):
            max_i = max(max_i,int(key.split('_')[1]))
    for i in range(0,max_i+1):
        img_str = request.files.get('image_'+str(i))
        img_sstor_to_pil = Image.open(img_str)
        images.append(img_sstor_to_pil)
    upscale_by = int(request.form.get('upscale_factor',4))
    risparmia_memoria = bool(request.form.get('risparmio_memoria',"False").capitalize())
    image_urls = []
    if risparmia_memoria:
        tile_size =500
    else:
        tile_size = 750
    images_upscaled = upscale(images,tile=tile_size,scale_factor=upscale_by,half_precision=risparmia_memoria, model_name="RealESRGAN_x2plus" if upscale_by == 2 else "RealESRGAN_x4plus")
    #save the generated images in immagini_generate/upscaled
    uninque_id = uuid.uuid4()
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    os.makedirs(os.path.join(img_dir_immagini_finali,'ingrandite',dt_string), exist_ok=True)
    counter=0
    for im in images_upscaled:
        counter+=1
        im.save('immagini_generate/ingrandite/'+dt_string+'/'+str(counter)+'.png')
        #append the url to the list of urls
        image_urls.append(url_for("mostra",filename=str(counter)+'.png',dirnew='ingrandite',subdir=dt_string))

    # Crea un buffer in memoria per il file zip
    zip_buffer = BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for index, image_url in enumerate(image_urls, start=1):
            # Assicurati che il percorso dell'immagine sia corretto
            image_path = os.path.join('immagini_generate', 'ingrandite', dt_string, f'{index}.png')
            zip_file.write(image_path, arcname=f'{index}.png')

    # Imposta il puntatore del buffer all'inizio del file
    zip_buffer.seek(0)


    return send_file(
        zip_buffer,
        as_attachment=True,
        mimetype='application/zip',
        download_name=f'immagini_generate_{dt_string}.zip'
    )
    #return the images with send_file





if __name__ == '__main__':
    #launchwith waitress
    from waitress import serve
    print("Starting server...")
    serve(app, host="0.0.0.0", port=5000)
    #app.run(debug=False, port=80,ssl_context='adhoc')




