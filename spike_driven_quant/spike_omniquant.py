import torch
import torch.nn as nn
from models.spike_llama_layer import QuantLlamaDecoderLayer
from models.int_opt_layer import QuantOPTDecoderLayer
from models.int_falcon_layer import QuantFalconDecoderLayer
from models.spike_mixtral_layer import SpikeMixtralDecoderLayer
from spike_driven_quant.spike_linear import SpikeQuantLinear
from spike_driven_quant.spike_matmul import SpikeQuantMatMul
from contextlib import nullcontext
import copy
import math
import utils
import os
import pdb
import gc
from spike_driven_quant.utils import let_parameters, lwc_parameters, get_omni_parameters,\
                            omni_state_dict, register_scales_and_zeros,smooth_and_quant_temporary,\
                            smooth_and_quant_inplace,clear_temp_variable,set_quant_state
try:
    import auto_gptq.nn_modules.qlinear.qlinear_cuda as qlinear_cuda
    import auto_gptq.nn_modules.qlinear.qlinear_triton as qlinear_triton
except:
    print("auto_gptq is required for real quantization")
import time # 添加导入



def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, SpikeQuantLinear)}


def add_new_module(name, original_module, added_module):
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = original_module
        for l_idx in range(len(levels)-1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], added_module)
    else:
        setattr(original_module, name, added_module)     

def find_layers(module, layers=[SpikeQuantLinear, SpikeQuantMatMul], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def static(layer, nsamples, inps, attention_mask, position_ids):
    print("Starting static calibration...")
    start_time = time.time() # 记录开始时间
    samples = nsamples
    subset = find_layers(layer)

    def add_batch(name):
        def tmp(_, inp, out):
            subset[name].add_batch(inp[0], out.data)
        return tmp

    handles = []
    res = None
    for name in subset:
        handles.append(subset[name].register_forward_hook(add_batch(name)))
    
    # 确保position_ids与inps在同一设备上
    device = inps[0].device
    if position_ids is not None:
        position_ids = position_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
        
    
    forward_pass_start_time = time.time()
    for j in range(samples):
        if (j + 1) % 10 == 0: # 每10次打印进度
            print(f"  Static calibration sample {j + 1}/{samples}, Time elapsed: {time.time() - forward_pass_start_time:.2f}s")
        try:
            res = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        except Exception as e:
            print(f"Error during static forward pass at sample {j}: {e}")
            raise e
    forward_pass_duration = time.time() - forward_pass_start_time
    

    for h in handles:
        h.remove()
    
    
    static_calc_start_time = time.time()
    for name in subset:
        
        subset[name].static()
    static_calc_duration = time.time() - static_calc_start_time
    
    total_duration = time.time() - start_time
    
    del subset


def spike_omniquant(
    lm,
    args,
    dataloader,
    act_scales,
    act_shifts,
    logger=None,
):
    logger.info("Starting ...")
    
    # 最简单解决方案：先将整个模型移到GPU
    logger.info("Moving the entire model to GPU for initial processing...")
    lm.model = lm.model.to(lm.device)
    
    # move embedding layer and first layer to target device
    model = lm.model
    dev = lm.device
    use_cache = model.config.use_cache
    model.config.use_cache = False
    is_llama = False
    if "llama" in args.net.lower():
        is_llama = True
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        DecoderLayer = QuantLlamaDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "o_proj":"out",
            "up_proj":"fc1"
        }
        layer_name_prefix = "model.layers"
    elif "opt" in args.net.lower():
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
        DecoderLayer = QuantOPTDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "out_proj":"out",
            "fc1":"fc1"
        }
        layer_name_prefix = "model.decoder.layers"
    elif "falcon" in args.net.lower():
        layers = model.transformer.h
        model.transformer.word_embeddings.to(dev)
        model.transformer.ln_f.to(dev)
        model.lm_head.to(dev)
        DecoderLayer = QuantFalconDecoderLayer
        layer_name_prefix = "model.transformer.h"
    elif 'mixtral' in args.net.lower():
        is_llama = True   # same to llama except ffn
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        DecoderLayer = SpikeMixtralDecoderLayer
        layer_name_prefix = "model.layers"
        pairs = {
            "q_proj":"qkv",
            "o_proj":"out",
            "w1":"fc1",
            "w2":"fc2",
            "w3":"fc3"
        }
    else:
        raise ValueError("Only support for opt/llama/Llama-2/falcon/mixtral now")
    
    
    # 显式移动第一层到GPU
    layers[0] = layers[0].to(dev)
    
    # 关键修改：显式检查并确保rotary embedding也移动到GPU
    if is_llama:
        # 检查第一层是否有rotary_emb，并确保它在正确的设备上
        if hasattr(layers[0], 'self_attn') and hasattr(layers[0].self_attn, 'rotary_emb'):
            # 打印设备以便诊断
            logger.info(f"Before explicit move - Rotary emb device: {layers[0].self_attn.rotary_emb.inv_freq.device}")
            # 显式移动rotary_emb到GPU
            layers[0].self_attn.rotary_emb = layers[0].self_attn.rotary_emb.to(dev)
            logger.info(f"After explicit move - Rotary emb device: {layers[0].self_attn.rotary_emb.inv_freq.device}")
    
    if args.deactive_amp and args.epochs>0:
        dtype = torch.float
        traincast = nullcontext
    else:
        dtype = torch.float16
        traincast = torch.cuda.amp.autocast
    inps = torch.zeros(
        (args.nsamples, lm.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0}

    # catch the first layer input
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.is_llama = False

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            
            # 保存所有传入的kwargs
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor):
                    cache[k] = v.to(inp.device)
            
            # 确保attention_mask被保存
            if "attention_mask" in kwargs:
                cache["attention_mask"] = kwargs["attention_mask"].to(inp.device)
            
            # 特别处理position_ids
            if self.is_llama:
                if "position_ids" in kwargs:
                    # 使用传入的position_ids
                    cache["position_ids"] = kwargs["position_ids"].to(inp.device)
                else:
                    # 如果没有传入，创建默认的position_ids
                    seq_len = inp.shape[1]  # 假设inp形状为[batch_size, seq_len, hidden_dim]
                    cache["position_ids"] = torch.arange(0, seq_len, dtype=torch.long, device=inp.device).unsqueeze(0)
            
            raise ValueError

    layers[0] = Catcher(layers[0])
    layers[0].is_llama = is_llama

    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                # 关键修改：完全替换transformers库默认生成的position_ids
                batch_tensor = batch[0].to(dev)
                batch_size, seq_len = batch_tensor.shape[0], batch_tensor.shape[1]
                
                # 创建明确的position_ids并确保在GPU上
                position_ids = torch.arange(0, seq_len, dtype=torch.long, device=dev).unsqueeze(0)
                position_ids = position_ids.expand(batch_size, -1)
                
                # 创建attention_mask (可选，取决于你的数据)
                attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=dev)
                
                # 确保所有参数都在同一设备上
                model(
                    batch_tensor,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False
                )
            except ValueError:
                pass
    
    # move embedding layer and first layer to cpu
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if "llama" in args.net.lower() or "mixtral" in args.net.lower():
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    elif "opt" in args.net.lower():
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    elif 'falcon' in args.model:
        model.transformer.word_embeddings =  model.transformer.word_embeddings.cpu()
    else:
        raise ValueError("Only support for opt/llama/Llama-2/falcon/mixtral now")
    torch.cuda.empty_cache()

    
    # same input of first layer for fp model and quant model
    quant_inps = inps
    fp_inps = copy.deepcopy(inps)   # take output of fp model as input
    fp_inps_2 = copy.deepcopy(inps) if args.aug_loss else None # take output of quantization model as input
    
    attention_mask = cache["attention_mask"]

    if attention_mask is not None:
        attention_mask_batch = attention_mask.repeat(args.batch_size,1,1,1) if args.deactive_amp else attention_mask.repeat(args.batch_size,1,1,1).float()
    else:
        logger.info(
            "No attention mask caught from the first layer."
            " Seems that model's attention works without a mask."
        )
        attention_mask_batch = None

    loss_func = torch.nn.MSELoss()
    if is_llama:
        position_ids = cache["position_ids"].to(dev)
    else:
        position_ids = None



    if args.resume:
        omni_parameters = torch.load(args.resume)
    else:
        omni_parameters = {}

    
    
    for i in range(len(layers)):
        logger.info(f"=== Start quantize layer {i} ===")
        layer = layers[i].to(dev)
        
        # 确保position_ids在GPU上且被明确传递
        if is_llama:
            position_ids = position_ids.to(dev)
        
        qlayer = DecoderLayer(lm.model.config, layer, args)
        qlayer = qlayer.to(dev)

        
        # obtain output of full-precision model
        set_quant_state(qlayer, weight_quant=False, act_quant=False)
        if args.epochs > 0:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    for j in range(args.nsamples):
                        # 确保使用的是当前设备
                        current_device = fp_inps[j].device
                        
                        # 1. 获取当前样本的形状
                        batch_size, seq_len = 1, fp_inps[j].shape[0]  # 单样本，所以batch_size=1
                        
                        # 2. 创建适当的position_ids（始终在当前设备上）
                        if is_llama:
                            current_position_ids = torch.arange(0, seq_len, dtype=torch.long, device=current_device).unsqueeze(0)
                        else:
                            current_position_ids = None
                        
                        # 3. 确保attention_mask也在当前设备上
                        current_attention_mask = None
                        if attention_mask is not None:
                            current_attention_mask = attention_mask.to(current_device)
                        
                        # 4. 调用模型时传递这些参数
                        fp_inps[j] = qlayer(
                            fp_inps[j].unsqueeze(0),
                            attention_mask=current_attention_mask,
                            position_ids=current_position_ids
                        )[0]
                        
                        # 对于aug_loss的情况也是类似处理
                        if args.aug_loss:
                            quant_device = quant_inps[j].device
                            if is_llama:
                                quant_position_ids = torch.arange(0, seq_len, dtype=torch.long, device=quant_device).unsqueeze(0)
                            else:
                                quant_position_ids = None
                            
                            quant_attention_mask = None
                            if attention_mask is not None:
                                quant_attention_mask = attention_mask.to(quant_device)
                                
                            fp_inps_2[j] = qlayer(
                                quant_inps[j].unsqueeze(0),
                                attention_mask=quant_attention_mask,
                                position_ids=quant_position_ids
                            )[0]

                    # 在static函数调用前再次确保position_ids在正确的设备上
                    if is_llama and position_ids is not None:
                        position_ids = position_ids.to(dev)
                    
                    # 在static函数调用前
                    # 获取quant_inps的设备（确保所有输入都在同一设备上）
                    static_device = quant_inps[0].device

                    # 创建适当的position_ids和attention_mask
                    if is_llama:
                        seq_len = quant_inps[0].shape[0]
                        static_position_ids = torch.arange(0, seq_len, dtype=torch.long, device=static_device).unsqueeze(0)
                    else:
                        static_position_ids = None

                    static_attention_mask = None
                    if attention_mask is not None:
                        static_attention_mask = attention_mask.to(static_device)

                    # 调用static函数
                    static(qlayer, args.nsamples, quant_inps, static_attention_mask, static_position_ids)
                    logger.info(f"--- Finished static calibration for layer {i} ---")


        # init smooth parameters
        set_quant_state(qlayer, weight_quant=False, act_quant=True)  # weight will be manually quantized before forward
        qlayer.let = args.let
        use_shift = True 
        if is_llama or args.abits == 16:
            use_shift = False                   # deactivate channel-wise shifting for llama model and weight-only quantization
        if args.let:
            # init channel-wise scaling and shift
            qlayer.register_parameter("qkt_smooth_scale",torch.nn.Parameter(torch.ones(layer.self_attn.q_proj.out_features,device=dev, dtype=dtype)))
            for name,module in qlayer.named_modules():
                if isinstance(module, SpikeQuantLinear):
                    for key in pairs.keys():
                        if key in name:
                            act = act_scales[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=dtype).clamp(min=1e-5)
                            weight = module.weight.abs().max(dim=0)[0].clamp(min=1e-5)
                            scale = (act.pow(args.alpha)/weight.pow(1-args.alpha)).clamp(min=1e-5)
                            if use_shift and not is_llama:
                                shift = act_shifts[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=dtype)
                            else:
                                shift = torch.zeros_like(scale)
                            qlayer.register_parameter(f"{pairs[key]}_smooth_shift",torch.nn.Parameter(shift))
                            qlayer.register_parameter(f"{pairs[key]}_smooth_scale",torch.nn.Parameter(scale))
                                
        if args.resume:
            qlayer.load_state_dict(omni_parameters[i], strict=False)
        

        if args.epochs > 0:
            with torch.no_grad():
                qlayer.float()      # required for AMP training
            # create optimizer
            optimizer = torch.optim.AdamW(
                [{"params":let_parameters(qlayer, use_shift),"lr":args.let_lr}, {"params":lwc_parameters(qlayer),"lr":args.lwc_lr}],weight_decay=args.wd)
            loss_scaler = utils.NativeScalerWithGradNormCount()
            
            for epochs in range(args.epochs):
                loss_list = []
                norm_list = []
                for j in range(args.nsamples//args.batch_size):    
                    # try:
                        index = j * args.batch_size
                        # obtain output of quantization model
                        with traincast():
                            smooth_and_quant_temporary(qlayer, args, is_llama)
                            quant_out = qlayer(quant_inps[index:index+args.batch_size,], attention_mask=attention_mask_batch,position_ids=position_ids)[0]
                            loss = loss_func(fp_inps[index:index+args.batch_size,], quant_out)
                            if args.aug_loss:
                                loss += loss_func(fp_inps_2[index:index+args.batch_size,], quant_out)
                        if not math.isfinite(loss.item()):
                            logger.info("Loss is NAN, stopping training")
                            pdb.set_trace()
                            
                        loss_list.append(loss.detach().cpu())
                        optimizer.zero_grad()
                        norm = loss_scaler(loss, optimizer,parameters= get_omni_parameters(qlayer, use_shift)).cpu()
                        norm_list.append(norm.data)
                    # except:
                    #     print("########### one false ###########")
                    #     pass

                loss_mean = torch.stack(loss_list).mean()
                norm_mean = torch.stack(norm_list).mean()
                logger.info(f"layer {i} iter {epochs} loss:{loss_mean} norm:{norm_mean} max memory_allocated {torch.cuda.max_memory_allocated(lm._device) / 1024**2} ")
            clear_temp_variable(qlayer)
            del optimizer
        qlayer.half() 
        # real smooth and quantization
        smooth_and_quant_inplace(qlayer, args, is_llama)
        if args.epochs>0:
            # update input of quantization model
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                # with traincast():
                    for j in range(args.nsamples):
                        quant_inps[j] = qlayer(quant_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]
            register_scales_and_zeros(qlayer)
            layers[i] = qlayer.to("cpu")
            omni_parameters[i] = omni_state_dict(qlayer)
            torch.save(omni_parameters, os.path.join(args.output_dir, f"omni_parameters.pth"))
        else:
            register_scales_and_zeros(qlayer)
            layers[i] = qlayer.to("cpu")
        if args.real_quant:
            assert args.wbits in [2,3,4] and args.abits >= 16   # only support weight-only quantization
            named_linears = get_named_linears(qlayer)
            for name, module in named_linears.items():
                scales = module.weight_quantizer.scales
                zeros = module.weight_quantizer.zeros
                group_size = module.weight_quantizer.group_size
                dim0 = module.weight.shape[0]
                scales = scales.view(dim0,-1)
                zeros = zeros.view(dim0,-1)
                if args.wbits == 3:
                    q_linear = qlinear_cuda.SpikeQuantLinear(args.wbits, group_size, module.in_features,module.out_features,not module.bias is None)
                else:
                    q_linear = qlinear_triton.SpikeQuantLinear(args.wbits, group_size, module.in_features,module.out_features,not module.bias is None)
                q_linear.pack(module.cpu(),  scales.float().cpu(), zeros.float().cpu())
                add_new_module(name, qlayer, q_linear)       
                print(f"pack quantized {name} finished")
                del module        
        del layer
        torch.cuda.empty_cache()

    del inps
    del quant_inps
    del fp_inps
    del fp_inps_2
    torch.cuda.empty_cache()
    gc.collect()                    
    model.config.use_cache = False
    return model

# 添加一个辅助函数来确保正确传递position_ids
def ensure_position_ids(batch, device):
    """为batch生成适当的position_ids并确保它在正确的设备上"""
    batch_size = batch.shape[0]
    seq_len = batch.shape[1]
    # 创建与batch相同大小的position_ids
    position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)
    return position_ids

def ensure_tensors_on_device(kwargs_dict, device):
    """确保字典中的所有张量都在指定设备上"""
    for k, v in kwargs_dict.items():
        if isinstance(v, torch.Tensor):
            kwargs_dict[k] = v.to(device)
    return kwargs_dict

