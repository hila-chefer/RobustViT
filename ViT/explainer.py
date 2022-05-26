import torch
import numpy as np
import cv2

# rule 5 from paper
def avg_heads(cam, grad):
    cam = cam.reshape(-1, cam.shape[-3], cam.shape[-2], cam.shape[-1])
    grad = grad.reshape(-1, cam.shape[-3], grad.shape[-2], grad.shape[-1])
    cam = grad * cam
    cam = cam.clamp(min=0).mean(dim=1)
    return cam

# rule 6 from paper
def apply_self_attention_rules(R_ss, cam_ss):
    R_ss_addition = torch.matmul(cam_ss, R_ss)
    return R_ss_addition

def upscale_relevance(relevance):
    relevance = relevance.reshape(-1, 1, 14, 14)
    relevance = torch.nn.functional.interpolate(relevance, scale_factor=16, mode='bilinear')

    # normalize between 0 and 1
    relevance = relevance.reshape(relevance.shape[0], -1)
    min = relevance.min(1, keepdim=True)[0]
    max = relevance.max(1, keepdim=True)[0]
    relevance = (relevance - min) / (max - min)

    relevance = relevance.reshape(-1, 1, 224, 224)
    return relevance

def generate_relevance(model, input, index=None):
    # a batch of samples
    batch_size = input.shape[0]
    output = model(input, register_hook=True)
    if index == None:
        index = np.argmax(output.cpu().data.numpy(), axis=-1)
        index = torch.tensor(index)

    one_hot = np.zeros((batch_size, output.shape[-1]), dtype=np.float32)
    one_hot[torch.arange(batch_size), index.data.cpu().numpy()] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.to(input.device) * output)
    model.zero_grad()

    num_tokens = model.blocks[0].attn.get_attention_map().shape[-1]
    R = torch.eye(num_tokens, num_tokens).cuda()
    R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    for i, blk in enumerate(model.blocks):
        grad = torch.autograd.grad(one_hot, [blk.attn.attention_map], retain_graph=True)[0]
        cam = blk.attn.get_attention_map()
        cam = avg_heads(cam, grad)
        R = R + apply_self_attention_rules(R, cam)
    relevance = R[:, 0, 1:]
    return upscale_relevance(relevance)

# create heatmap from mask on image
def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam


def get_image_with_relevance(image, relevance):
    image = image.permute(1, 2, 0)
    relevance = relevance.permute(1, 2, 0)
    image = (image - image.min()) / (image.max() - image.min())
    image = 255 * image
    vis = image * relevance
    return vis.data.cpu().numpy()