from peft import PeftModel, PeftConfig
from transformers import AutoConfig, AutoModelForCausalLM
import torch

def load_ft_model(model_dir: str, **kwargs):
    # 加载 PEFT 配置
    peft_config = PeftConfig.from_pretrained(model_dir)
    
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        torch_dtype=torch.bfloat16,
        **kwargs
    )
    
    # 加载 LoRA adapter
    model = PeftModel.from_pretrained(base_model, model_dir)
    return model


def load_model(model_dir: str, state_dict=None, device_map=None):
    config = AutoConfig.from_pretrained(model_dir)

    model = AutoModelForCausalLM.from_config(config)
    if state_dict is not None:
        model.load_state_dict(state_dict)

    if device_map == "auto":
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")  # or use accelerate

    return model

# def load_model(model_dir: str, **kwargs) -> AutoModelForCausalLM:
#     return AutoModelForCausalLM.from_pretrained(
#         model_dir,
#         torch_dtype=torch.bfloat16,
#         **kwargs
#     )


def compare(model1, model2) -> bool:
    """Compares two models.

    Args:
        model1 (_type_): _description_
        model2 (_type_): _description_

    Returns:
        bool: _description_
    """
    dict1, dict2 = model1.state_dict(), model2.state_dict()
    if dict1.keys() != dict2.keys():
        return False
    for key in dict1.keys():
        if not torch.equal(dict1[key], dict2[key]):
            return False
    return True


def unlearn(
    model_dir: str,
    out_dir: str | None = None,
    some_pt_model_dir: str | None = None,
    some_ft_model_dir: str | None = None,
    alpha: float = 1.0
):
    if some_pt_model_dir is None or some_ft_model_dir is None:
        raise ValueError("Task vector (ilharco2023) requires some pretrained & finetuned models!")
    
    # 加载 finetuned 的 LoRA 模型
    finetuned_model = load_ft_model(some_ft_model_dir)
    merged_finetuned_model = finetuned_model.merge_and_unload()

    # 加载预训练模型（普通模型即可）
    pretrained_model = AutoModelForCausalLM.from_pretrained(
        some_pt_model_dir,
        torch_dtype=torch.bfloat16
    )

    # 计算 task vector
    task_vector = TaskVector(
        pretrained_state_dict=pretrained_model.state_dict(),
        finetuned_state_dict=merged_finetuned_model.state_dict()
    )

    if not task_vector.is_nonzero():
        raise ValueError("Zero task vector encountered!")

    neg_task_vector = -task_vector

    model = load_model(model_dir)
    new_state_dict = neg_task_vector.apply_to(pretrained_model=model, scaling_coef=alpha, in_place=False)
    del model
    new_model = load_model(model_dir, state_dict=new_state_dict, device_map='auto')

    if out_dir is not None:
        new_model.save_pretrained(out_dir)
    return new_model


class TaskVector():
    def __init__(self,
                 pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None,
                 pretrained_state_dict=None, finetuned_state_dict=None):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.
        
        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passying in
        the task vector state dict.
        """
        if vector is not None:
            self.vector = vector
        else:
            assert (
                (pretrained_checkpoint is not None and finetuned_checkpoint is not None)
                or
                (pretrained_state_dict is not None and finetuned_state_dict is not None)
            )
            with torch.no_grad():
                if pretrained_state_dict is None:
                    pretrained_state_dict = torch.load(pretrained_checkpoint).state_dict()
                if finetuned_state_dict is None:
                    finetuned_state_dict = torch.load(finetuned_checkpoint).state_dict()
                self.vector = {}
                for key in pretrained_state_dict:
                    if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                        continue
                    self.vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]

    
    def __add__(self, other):
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f'Warning, key {key} is not present in both task vectors.')
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return TaskVector(vector=new_vector)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = - self.vector[key]
        return TaskVector(vector=new_vector)

    def is_nonzero(self):
        return any([(self.vector[key] != 0).any() for key in self.vector])

    def apply_to(self, pretrained_model, scaling_coef=1.0, in_place=False):
        """Apply a task vector to a pretrained model."""
        with torch.no_grad():
            new_state_dict = {}
            pretrained_state_dict = pretrained_model.state_dict()
            for key in pretrained_state_dict:
                if key not in self.vector:
                    print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                    continue
                new_state_dict[key] = pretrained_state_dict[key] + scaling_coef * self.vector[key]
        if in_place:
            pretrained_model.load_state_dict(new_state_dict, strict=False)
        return new_state_dict
