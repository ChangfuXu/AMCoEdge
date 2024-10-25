import random
import einops
from pytorch_lightning import seed_everything
from ldm_hacked import *


def image_repair(image_, mask_, num_samples_):
    # config的地址
    config_path = "model_data/sd_v15.yaml"
    # 模型的地址
    model_path = "model_data/v1-5-pruned-emaonly.safetensors"

    sd_fp16 = True  # fp16，可以加速与节省显存
    vae_fp16 = True

    ddim_steps = 20  # 采样的步数
    seed = 12345  # 采样的种子，为-1的话则随机
    eta = 0  # eta
    denoise_strength = 1.00  # denoise强度
    scale = 9  # 正负扩大倍数

    # 提示词
    prompt = "a human face"
    # 正面提示词
    a_prompt = "best quality, extremely detailed"
    # 负面提示词
    n_prompt = "cropped, worst quality, low quality"

    #  创建模型
    model = create_model(config_path).cpu()
    model.load_state_dict(load_state_dict(model_path, location='cuda'), strict=False)
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)
    if sd_fp16:
        model = model.half()

    with torch.no_grad():
        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        #  对输入图片进行编码并加噪
        image_ = HWC3(np.array(image_, np.uint8))
        image_ = torch.from_numpy(image_.copy()).float().cuda() / 127.0 - 1.0
        image_ = torch.stack([image_ for _ in range(num_samples_)], dim=0)
        image_ = einops.rearrange(image_, 'b h w c -> b c h w').clone()
        if vae_fp16:
            image_ = image_.half()
            model.first_stage_model = model.first_stage_model.half()
        else:
            model.first_stage_model = model.first_stage_model.float()

        ddim_sampler.make_schedule(ddim_steps, ddim_eta=eta, verbose=True)
        t_enc = min(int(denoise_strength * ddim_steps), ddim_steps - 1)
        # 获得VAE编码后的隐含层向量
        z = model.get_first_stage_encoding(model.encode_first_stage(image_))
        x0 = z

        # 获得加噪后的隐含层向量
        z_enc = ddim_sampler.stochastic_encode(z, torch.tensor([t_enc] * num_samples_).to(model.device))
        z_enc = z_enc.half() if sd_fp16 else z_enc.float()

        # Achieve the latent vector of mask
        mask_ = torch.from_numpy(mask_).to(model.device)
        mask_ = torch.nn.functional.interpolate(mask_, size=z_enc.shape[-2:])

        # Encoding for the prompt
        cond = {"c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples_)]}
        un_cond = {"c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples_)]}

        # Do image repairing
        samples = ddim_sampler.decode(z_enc, cond, t_enc, mask_, x0, unconditional_guidance_scale=scale,
                                      unconditional_conditioning=un_cond)
        #  Decoding for samples
        x_samples = model.decode_first_stage(samples.half() if vae_fp16 else samples.float())
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c')
                     * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
    return x_samples


if __name__ == '__main__':
    image_path = "input/face.jpg"  # The original image path
    mask_path = "input/thin_mask.png"  # The path of mask image
    save_path = "output"  # The path of saving image
    input_shape = [512, 512]  # Set the shape of input image

    image = Image.open(image_path)
    image = crop_and_resize(image, input_shape[0], input_shape[1])

    mask = Image.open(mask_path).convert("L")
    mask = crop_and_resize(mask, input_shape[0], input_shape[1])
    mask = np.array(mask)
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1

    num_samples = 3  # The number of repaired image samples
    results = image_repair(image, mask, num_samples)
    # save repaired images
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for index, image in enumerate(results):
        cv2.imwrite(os.path.join(save_path, "result" + str(index) + ".jpg"), cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    #  save corrupted image
    corrupted_image = image_with_mask(image_path, mask_path)
    cv2.imwrite(os.path.join(save_path, "corrupted_image.jpg"), corrupted_image)
