# Strategy Overview

Understanding training strategies in ModelForge.

## What Are Strategies?

**Strategies** define how models are trained:
- Model preparation (adapters, PEFT configuration)
- Dataset formatting
- Trainer setup
- Training algorithm

Different strategies offer different trade-offs in terms of memory, speed, and quality.

## Available Strategies

| Strategy | Memory | Speed | Quality | Use Case |
|----------|--------|-------|---------|----------|
| **[SFT](sft.md)** | Baseline | 1x | High | General-purpose fine-tuning |
| **[QLoRA](qlora.md)** | -30-50% | 0.9x | High | Limited VRAM |
| **[RLHF](rlhf.md)** | Medium | Medium | Very High | Alignment with human preferences |
| **[DPO](dpo.md)** | Medium | Medium | Very High | Simpler alternative to RLHF |

## Choosing a Strategy

### Use SFT When:

✅ First time fine-tuning  
✅ Have sufficient VRAM  
✅ Standard supervised learning task  
✅ Want simplest setup  

### Use QLoRA When:

✅ Limited VRAM (< 12GB for 7B models)  
✅ Want to train larger models  
✅ Memory is the bottleneck  
✅ Can accept slightly slower training  

### Use RLHF When:

✅ Aligning model with human preferences
✅ Have preference pairs (prompt/chosen/rejected)
✅ Quality is critical
✅ Want conservative training defaults

### Use DPO When:

✅ Have preference pairs (chosen/rejected)  
✅ Want simpler alternative to RLHF  
✅ Alignment without reward model  
✅ More stable training than RLHF  

## Configuration

Specify strategy in training config:

```json
{
  "strategy": "sft"  // or "qlora", "rlhf", "dpo"
}
```

> **Note**: DPO and RLHF strategies require `"task": "text-generation"`. This is enforced by schema validation.

## Next Steps

- **[SFT Strategy](sft.md)** - Standard supervised fine-tuning
- **[QLoRA Strategy](qlora.md)** - Memory-efficient training
- **[Configuration Guide](../configuration/configuration-guide.md)** - All options

---

**Choose the right strategy for your needs!** 🎯
